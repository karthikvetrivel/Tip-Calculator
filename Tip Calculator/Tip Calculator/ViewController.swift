//
//  ViewController.swift
//  Tip Calculator
//
//  Created by Karsick on 3/8/18.
//  Copyright © 2018 Karthik Vetrivel. All rights reserved.
//

import UIKit


class ViewController: UIViewController {

    @IBOutlet weak var billView: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        billView.layer.cornerRadius = 20
        
        billView.layer.shadowColor = UIColor.blue.cgColor
        billView.layer.shadowOffset = CGSize(width: 3, height: 6)
        billView.layer.shadowOpacity = 1
        billView.layer.shadowRadius = 4.0
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
   
}

