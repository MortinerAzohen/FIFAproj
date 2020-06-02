from functools import partial
from tkinter import *
from Predictions.train_test_split_tkapp import TrainTestTkkApp


root = Tk()
root.title("FIFA_APP")
machineLearningObj = TrainTestTkkApp()

e = Entry(root, width=100, borderwidth = 5,state = 'readonly')
e.grid(row=0, column=0, columnspan = 5, padx=10, pady=10)

def show_price():
    data = [[entry_1.get(),entry_2.get(),entry_3.get(),entry_4.get(),entry_5.get(),entry_6.get(),entry_7.get(),entry_8.get(),entry_9.get(),entry_10.get(),entry_11.get(),entry_12.get()]]
    machineLearningObj.checkNewFifaPlayer(data)


def button_show(number):
    e.config(state='normal')
    e.delete(0,END)
    if(number==1):
        a = machineLearningObj.linearReg()
        string = "Mean squared error: {:.0f} Coefficient of determination: {:.02f}".format(a[0],a[1])
        e.insert(0,string)
        e.config(state='readonly')
    elif(number==2):
        a = machineLearningObj.decisionTree()
        string = "Mean squared error: {:.0f} Coefficient of determination: {:.02f}".format(a[0],a[1])
        e.insert(0,string)
        e.config(state='readonly')
    elif (number == 3):
        a = machineLearningObj.randomForest()
        string = "Mean squared error: {:.0f} Coefficient of determination: {:.02f}".format(a[0],a[1])
        e.insert(0,string)
        e.config(state='readonly')
    elif (number == 4):
        a = machineLearningObj.gradientBoost()
        string = "Mean squared error: {:.0f} Coefficient of determination: {:.02f}".format(a[0],a[1])
        e.insert(0,string)
        e.config(state='readonly')
    elif (number == 5):
        a = machineLearningObj.extraTree()
        string = "Mean squared error: {:.0f} Coefficient of determination: {:.02f}".format(a[0],a[1])
        e.insert(0,string)
        e.config(state='readonly')


btn_1=Button(root,text="Linear Regression",padx=40,pady=20,command=partial(button_show,1))
btn_2=Button(root,text="Decision Tree",padx=40,pady=20,command=partial(button_show,2))
btn_3=Button(root,text="Forest Regressior",padx=40,pady=20,command=partial(button_show,3))
btn_4=Button(root,text="Gradient Boosting",padx=40,pady=20,command=partial(button_show,4))
btn_5=Button(root,text="Extra Trees Regressor",padx=40,pady=20,command=partial(button_show,5))
getPrice = Button(root,text="SHOW PRICE",padx=40,pady=10,command=show_price)
entry_1 = Entry(root, width=10, borderwidth = 5,)
entry_2 = Entry(root, width=10, borderwidth = 5,)
entry_3 = Entry(root, width=10, borderwidth = 5,)
entry_4 = Entry(root, width=10, borderwidth = 5,)
entry_5 = Entry(root, width=10, borderwidth = 5,)
entry_6 = Entry(root, width=10, borderwidth = 5,)
entry_7 = Entry(root, width=10, borderwidth = 5,)
entry_8 = Entry(root, width=10, borderwidth = 5,)
entry_9 = Entry(root, width=10, borderwidth = 5,)
entry_10 = Entry(root, width=10, borderwidth = 5,)
entry_11 = Entry(root, width=10, borderwidth = 5,)
entry_12 = Entry(root, width=10, borderwidth = 5,)

entry_1.grid(row=3, column=0, columnspan = 1, padx=10, pady=10)
entry_2.grid(row=3, column=1, columnspan = 1, padx=10, pady=10)
entry_3.grid(row=3, column=2, columnspan = 1, padx=10, pady=10)
entry_4.grid(row=3, column=3, columnspan = 1, padx=10, pady=10)
entry_5.grid(row=3, column=4, columnspan = 1, padx=10, pady=10)
entry_6.grid(row=4, column=0, columnspan = 1, padx=10, pady=10)
entry_7.grid(row=4, column=1, columnspan = 1, padx=10, pady=10)
entry_8.grid(row=4, column=2, columnspan = 1, padx=10, pady=10)
entry_9.grid(row=4, column=3, columnspan = 1, padx=10, pady=10)
entry_10.grid(row=4, column=4, columnspan = 1, padx=10, pady=10)
entry_11.grid(row=5, column=0, columnspan = 1, padx=10, pady=10)
entry_12.grid(row=5, column=1, columnspan = 1, padx=10, pady=10)
btn_1.grid(row=1,column=0)
btn_2.grid(row=1,column=1)
btn_3.grid(row=1,column=2)
btn_4.grid(row=1,column=3)
btn_5.grid(row=1,column=4)
getPrice.grid(row=5, column = 2)


root.mainloop()