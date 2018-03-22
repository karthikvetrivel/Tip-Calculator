//
//  SettingsController.swift
//  Tip Calculator
//
//  Created by Adrian Avram on 3/17/18.
//  Copyright © 2018 Karthik Vetrivel. All rights reserved.
//

import UIKit

class SettingsController: UITableViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        self.title = "Settings"
    }
    
//    override func viewDidAppear(_ animated: Bool) {
//        self.navigationController?.setNavigationBarHidden(false, animated: true)
//    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
    }
}
