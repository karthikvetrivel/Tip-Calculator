//
//  ViewController.swift
//  Tip Calculator
//
//  Created by Karsick on 3/8/18.
//  Copyright © 2018 Karthik Vetrivel. All rights reserved.
//

import UIKit


class ViewController: UIViewController {

    @IBOutlet weak var billTotal: UILabel!
    
    @IBOutlet weak var firstHeight: NSLayoutConstraint!
    
    @IBOutlet weak var purpleHeight: NSLayoutConstraint!
    
    var totalBillAmount : Float = 0.0
    var decimalAmount : Float = 0.0
    var decimalLocation : Int = 1
    var decimal = false

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        // scaling some of the views.
       firstHeight.constant = (self.view.frame.size.width / 3)
        
       purpleHeight.constant = (self.view.frame.size.height / 3) - 30
        
    }

    @IBAction func buttonPressed(_ sender: UIButton) {
        
        switch sender.tag {
        // corresponds to each #, case 10 is the 0 button.
        case 1 ... 10:
            
            if !decimal {
                // adds numbers by the tens, hundereds
                totalBillAmount *= 10
                if sender.tag != 10{
                    totalBillAmount += Float(sender.tag)
                }
            } else {
                // starts adding numbers through the tenths, hundereths, etc.
                decimalAmount = Float(sender.tag)
                for _ in 0 ... decimalLocation-1 {
                    decimalAmount /= 10
                }
                
                decimalLocation += 1
                print(decimalAmount)
                totalBillAmount += decimalAmount
            }
        
        // the decimal button
        case 11:
            // switches the decimal system, seen above
            decimal = true
            
        // clear button
        case 12:
            // resets to the previous settings.
            decimal = false
            decimalLocation = 1
            totalBillAmount = 0.0
        default:
            print("An error occured")
        }
        
        // display billTotal
        formatBill(billAmount: totalBillAmount)
    }
    
    // update billTotal
    func formatBill( billAmount : Float) {
        billTotal.text = "$" + String(format: "%.2f", billAmount)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
   
}

