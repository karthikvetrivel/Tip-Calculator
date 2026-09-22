"""A small, loopback-only HTTP server for the read-only dashboard."""

from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import socket
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .code import build_code_file, build_pr_code_file, list_code_files, list_pr_code_files
from .report import build_dashboard


_STATIC_ROOT = Path(__file__).with_name("static")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/code": ("code.html", "text/html; charset=utf-8"),
    "/code.js": ("code.js", "text/javascript; charset=utf-8"),
    "/code.css": ("code.css", "text/css; charset=utf-8"),
    "/mona-sans-variable.woff2": ("mona-sans-variable.woff2", "font/woff2"),
    "/monaspace-neon-regular.woff2": ("monaspace-neon-regular.woff2", "font/woff2"),
}
_MAX_REQUEST_LINE = 4_096
_MAX_HEADER_BYTES = 16_384
_MAX_HEADER_COUNT = 64
_MAX_STATIC_BYTES = 2 * 1024 * 1024
# Match the GitHub footer's repository validator. These values are display-only
# route identity; neither is used as a filesystem path or Git argument.
_PR_SEGMENT = r"(?!\.{1,2}(?:/|$))[A-Za-z0-9_.-]{1,100}"
_PR_OID = r"[0-9a-f]{40}(?:[0-9a-f]{24})?"
_PR_ROUTE = re.compile(
    rf"^/pr/(?P<owner>{_PR_SEGMENT})/(?P<repository>{_PR_SEGMENT})/"
    rf"(?P<number>[1-9][0-9]{{0,8}})/(?P<base>{_PR_OID})/(?P<head>{_PR_OID})$"
)


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _LoopbackServerV6(_LoopbackServer):
    address_family = socket.AF_INET6


class _DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "JoyrideHTTP/1.0"
    sys_version = ""

    def __init__(
        self,
        *args: Any,
        repo: Path,
        target_ref: str,
        allowed_hosts: set[str],
        server_port: int,
        **kwargs: Any,
    ) -> None:
        self._repo = repo
        self._target_ref = target_ref
        self._allowed_hosts = allowed_hosts
        self._server_port = server_port
        super().__init__(*args, **kwargs)

    def version_string(self) -> str:
        return self.server_version

    def parse_request(self) -> bool:
        if len(self.raw_requestline) > _MAX_REQUEST_LINE:
            self.requestline = ""
            self.request_version = "HTTP/1.1"
            self.command = None
            self.send_error(414, "Request target is too long.")
            return False
        if not super().parse_request():
            return False
        if len(self.headers) > _MAX_HEADER_COUNT:
            self.send_error(431, "Too many request headers.")
            return False
        header_bytes = sum(
            len(name.encode("utf-8", errors="replace"))
            + len(value.encode("utf-8", errors="replace"))
            + 4
            for name, value in self.headers.items()
        )
        if header_bytes > _MAX_HEADER_BYTES:
            self.send_error(431, "Request headers are too large.")
            return False
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding is not None:
            self.close_connection = True
            self.send_error(400, "Request bodies are not accepted.")
            return False
        content_length = self.headers.get("Content-Length")
        if content_length is not None:
            try:
                body_length = int(content_length)
            except ValueError:
                self.close_connection = True
                self.send_error(400, "Invalid Content-Length header.")
                return False
            if body_length != 0:
                self.close_connection = True
                self.send_error(413, "Request bodies are not accepted.")
                return False
        return True

    def _security_headers(self) -> None:
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Cache-Control", "no-store")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        head_only: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        head_only: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            body = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            status = 500
            body = b'{"error":"Dashboard data could not be encoded."}'
        self._send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            head_only=head_only,
            extra_headers=extra_headers,
        )

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        del explain
        readable = message or self.responses.get(code, ("Request failed",))[0]
        self._send_json(
            code,
            {"error": readable},
            head_only=getattr(self, "command", None) == "HEAD",
        )

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the local dashboard quiet during normal browser polling."""

        del format, args

    def _valid_host(self) -> bool:
        host_headers = self.headers.get_all("Host", [])
        if len(host_headers) != 1:
            return False
        raw = host_headers[0]
        if (
            not raw
            or any(character.isspace() or ord(character) < 32 for character in raw)
            or "/" in raw
            or "\\" in raw
            or "@" in raw
        ):
            return False
        try:
            parsed = urlsplit(f"//{raw}")
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return False
        if (
            hostname is None
            or hostname.rstrip(".").lower() not in self._allowed_hosts
            or (port is not None and port != self._server_port)
        ):
            return False
        return True

    def _valid_peer(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False

    def _authorise_request(self) -> bool:
        if not self._valid_peer():
            self.send_error(403, "Only loopback clients are allowed.")
            return False
        if not self._valid_host():
            self.send_error(421, "Unrecognized Host header.")
            return False
        return True

    def _static_path(self) -> tuple[Path, str] | None:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            return None
        if _PR_ROUTE.fullmatch(parsed.path):
            return _STATIC_ROOT / "code.html", "text/html; charset=utf-8"
        item = _STATIC_FILES.get(parsed.path)
        if item is None:
            return None
        filename, content_type = item
        return _STATIC_ROOT / filename, content_type

    def _serve_static(self, *, head_only: bool) -> None:
        item = self._static_path()
        if item is None:
            self.send_error(404, "Dashboard route not found.")
            return
        path, content_type = item
        try:
            size = path.stat().st_size
            if size > _MAX_STATIC_BYTES:
                raise ValueError("asset exceeds the size limit")
            body = path.read_bytes()
        except (OSError, ValueError):
            self.send_error(500, "Dashboard asset is unavailable.")
            return
        self._send_bytes(200, body, content_type, head_only=head_only)

    def do_GET(self) -> None:
        if not self._authorise_request():
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/dashboard" and not parsed.query and not parsed.fragment:
            try:
                payload = build_dashboard(self._repo, target_ref=self._target_ref)
            except Exception as exc:
                message = str(exc).strip() or "Dashboard data is unavailable."
                self._send_json(500, {"error": message})
                return
            self._send_json(200, payload)
            return
        if parsed.path in {"/api/code/files", "/api/code/file"}:
            self._serve_code_api(parsed.path, parsed.query, parsed.fragment)
            return
        pr_api = self._pr_api_route(parsed.path)
        if pr_api is not None:
            self._serve_pr_code_api(pr_api, parsed.query, parsed.fragment)
            return
        self._serve_static(head_only=False)

    @staticmethod
    def _pr_api_route(path: str) -> tuple[dict[str, str], str] | None:
        if not path.startswith("/api/pr/"):
            return None
        for suffix in ("/files", "/file"):
            if not path.endswith(suffix):
                continue
            matched = _PR_ROUTE.fullmatch(path[4 : -len(suffix)])
            if matched:
                return matched.groupdict(), suffix[1:]
        return None

    def _serve_pr_code_api(
        self, route: tuple[dict[str, str], str], query: str, fragment: str
    ) -> None:
        details, action = route
        try:
            if fragment:
                raise ValueError("URL fragments are not accepted in API requests.")
            common = {
                "owner": details["owner"], "repository": details["repository"],
                "number": int(details["number"]), "target_ref": self._target_ref,
            }
            if action == "files":
                if query:
                    raise ValueError("The PR file list does not accept query parameters.")
                payload = list_pr_code_files(
                    self._repo, details["base"], details["head"], **common,
                )
            else:
                parameters = parse_qs(
                    query, keep_blank_values=True, strict_parsing=True,
                    encoding="utf-8", errors="strict", max_num_fields=2,
                )
                if set(parameters) - {"path", "revision"} or "path" not in parameters or any(len(values) != 1 or not values[0] for values in parameters.values()):
                    raise ValueError("A single repository-relative file path is required.")
                payload = build_pr_code_file(
                    self._repo, parameters["path"][0], details["base"], details["head"],
                    revision=parameters.get("revision", [None])[0], **common,
                )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc) or "Invalid PR code request."})
            return
        except Exception:
            self._send_json(500, {"error": "Pinned PR history is unavailable. Try Refresh."})
            return
        self._send_json(200, payload)

    def _serve_code_api(self, path: str, query: str, fragment: str) -> None:
        try:
            if fragment:
                raise ValueError("URL fragments are not accepted in API requests.")
            if path == "/api/code/files":
                if query:
                    raise ValueError("The file list does not accept query parameters.")
                payload = list_code_files(self._repo, target_ref=self._target_ref)
            else:
                parameters = parse_qs(
                    query, keep_blank_values=True, strict_parsing=True,
                    encoding="utf-8", errors="strict", max_num_fields=2,
                )
                if set(parameters) - {"path", "revision"}:
                    raise ValueError("Only path and revision parameters are accepted.")
                if any(len(values) != 1 or not values[0] for values in parameters.values()):
                    raise ValueError("Each parameter needs one non-empty value.")
                if "path" not in parameters:
                    raise ValueError("A repository-relative file path is required.")
                payload = build_code_file(
                    self._repo,
                    parameters["path"][0],
                    target_ref=self._target_ref,
                    revision=parameters.get("revision", [None])[0],
                )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc) or "Invalid code request."})
            return
        except Exception:
            self._send_json(500, {"error": "Code history is unavailable. Try Refresh."})
            return
        self._send_json(200, payload)

    def do_HEAD(self) -> None:
        if not self._authorise_request():
            return
        parsed = urlsplit(self.path)
        if parsed.path in {"/api/dashboard", "/api/code/files", "/api/code/file"} or self._pr_api_route(parsed.path) is not None:
            self._send_json(
                405,
                {"error": "HEAD is not supported for this route."},
                head_only=True,
                extra_headers={"Allow": "GET"},
            )
            return
        self._serve_static(head_only=True)

    def _method_not_allowed(self) -> None:
        if not self._authorise_request():
            return
        self._send_json(
            405,
            {"error": f"{self.command} is not supported."},
            extra_headers={"Allow": "GET, HEAD"},
        )

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_TRACE = _method_not_allowed
    do_CONNECT = _method_not_allowed


def _make_server(
    repo: str | Path,
    host: str,
    port: int,
    target_ref: str,
) -> ThreadingHTTPServer:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("The dashboard host must be a loopback IP address.") from exc
    if not address.is_loopback:
        raise ValueError("The dashboard can bind only to a loopback address.")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer between 0 and 65535.")
    if not isinstance(target_ref, str) or not target_ref or "\x00" in target_ref:
        raise ValueError("Target ref must be a non-empty string.")

    server_class = _LoopbackServerV6 if address.version == 6 else _LoopbackServer
    # Bind before constructing the handler so an ephemeral port can be included in
    # the exact Host-header allowlist.
    server = server_class((host, port), None, bind_and_activate=False)
    try:
        server.server_bind()
        server.server_activate()
        actual_port = int(server.server_address[1])
        allowed_hosts = {host.lower(), "localhost"}
        handler = partial(
            _DashboardHandler,
            repo=Path(repo).expanduser().resolve(),
            target_ref=target_ref,
            allowed_hosts=allowed_hosts,
            server_port=actual_port,
        )
        server.RequestHandlerClass = handler
    except BaseException:
        server.server_close()
        raise
    return server


def serve(
    repo: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    target_ref: str = "main",
    *,
    start_path: str = "/",
    open_browser: bool = False,
) -> None:
    """Serve the dashboard until interrupted, bound only to a loopback address."""

    if start_path not in {"/", "/code"} and not start_path.startswith("/code#"):
        raise ValueError("The start page must be the dashboard or code view.")
    with _make_server(repo, host, port, target_ref) as server:
        actual_host = str(server.server_address[0])
        actual_port = int(server.server_address[1])
        display_host = f"[{actual_host}]" if ":" in actual_host else actual_host
        url = f"http://{display_host}:{actual_port}{'' if start_path == '/' else start_path}"
        print(url, flush=True)
        if open_browser:
            threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
        server.serve_forever()


def _open_browser(url: str) -> None:
    try:
        opened = webbrowser.open(url)
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        print("Open the local URL above in your browser.", file=sys.stderr, flush=True)
