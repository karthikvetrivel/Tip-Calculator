//
//  ViewController.swift
//  Tip Calculator
//
//  Created by Karsick on 3/8/18.
//  Copyright © 2018 Karthik Vetrivel. All rights reserved.
//

import UIKit


class ViewController: UIViewController {

    @IBOutlet weak var tipTextField: UITextField!
    @IBOutlet weak var billTotal: UILabel!
    
    @IBOutlet weak var firstHeight: NSLayoutConstraint!
    
    @IBOutlet weak var purpleHeight: NSLayoutConstraint!
    
    var totalDecimal : Float = 0;
    var totalBillAmount : Float = 0.0
    var decimalAmount : Float = 0.0
    var decimalLocation : Int = 1
    var decimal = false
    var intBillAmount: Int = 0;
    var amountArr = [Int]();
    
    var selectedTip : Int = 25
    let tipAmount = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        // scaling some of the views.
        makePickerView()
        makeToolBar()
        
        firstHeight.constant = (self.view.frame.size.width / 3) - 10
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
                if sender.tag != 10 {
                    decimalAmount = Float(sender.tag)
                    for _ in 0 ... decimalLocation-1 {
                        decimalAmount /= 10
                    }
                   
                    totalDecimal += decimalAmount
                    // for the delete button
                    
                    decimalLocation += 1
                    totalBillAmount += decimalAmount
                }
              
        
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
            totalDecimal = 0.0;

            
        // delete button
        case 13:
            // only works with non-decimal numbers.
            if !decimal {
                turnToArray(amount: totalBillAmount)
                amountArr.popLast()
                turnToNumber()
                totalBillAmount = Float(intBillAmount)
                
            }  else {
                if decimalLocation == 3 {
                    totalDecimal *= 100;
                }
                
                if decimalLocation == 2 {
                    totalDecimal *= 10
                }
                
                let toTurnToInt = String(totalDecimal)
                amountArr = toTurnToInt.flatMap{Int(String($0))}
                amountArr.removeFirst()
                turnToNumber()
                totalDecimal = Float(intBillAmount)
                totalBillAmount -= (totalDecimal/1000)
                
            }
            
            
            
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
    
    
    func turnToArray(amount: Float) {
        let toTurnToInt = String(Int(amount))
        amountArr = toTurnToInt.flatMap{Int(String($0))} // [1, 2, 3, 4, 5, 6]
        // turns the number in a string and removes the last value.
        
    }
    
    func turnToNumber() {
        intBillAmount = 0;
        var i : Int = 0;
        for _ in amountArr {
            intBillAmount *= 10
            intBillAmount += amountArr[i];
            i += 1
        }
    }
    // turn the array back to #'s
    
    func makePickerView() {
        
        let pickerView = UIPickerView()
        pickerView.delegate = self
        tipTextField.inputView = pickerView
        
        // Customize color here
        pickerView.backgroundColor = UIColor(red:0.22, green:0.23, blue:0.31, alpha:1.0)
    }
    
    func makeToolBar() {
        
        let toolBar = UIToolbar()
        toolBar.barStyle = UIBarStyle.default
        toolBar.isTranslucent = true
        toolBar.sizeToFit()
        
        let doneButton = UIBarButtonItem(title: "Done", style: UIBarButtonItemStyle.plain, target: self, action: #selector(ViewController.dismissKeyboard))
        
        toolBar.setItems([doneButton], animated: false)
        toolBar.isUserInteractionEnabled = true
        
        // Customize color here
        toolBar.barTintColor = UIColor(red:0.17, green:0.18, blue:0.28, alpha:1.0)
        tipTextField.inputAccessoryView = toolBar
        
    }
    
    @objc func dismissKeyboard() {
        self.view.endEditing(true)
    }
  
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
   
}

extension ViewController: UIPickerViewDelegate, UIPickerViewDataSource{
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return tipAmount.count
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return String(tipAmount[row])
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        selectedTip = tipAmount[row]
        tipTextField.text = "     " + String(selectedTip)
    }
    
    func pickerView(_ pickerView: UIPickerView, attributedTitleForRow row: Int, forComponent component: Int) -> NSAttributedString? {
        
        let string = String(tipAmount[row])
        return NSAttributedString(string: string, attributes: [NSAttributedStringKey.foregroundColor: UIColor.white])
    }
    
    
    

}

