from functools import partial
from tkinter import *
from Predictions.train_test_split_tkapp import TrainTestTkkApp
from tkinter.ttk import Combobox


class Lotfi(Entry):
    def __init__(self, master=None, **kwargs):
        self.var = StringVar()
        Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.old_value = ''
        self.var.trace('w', self.check)
        self.get, self.set = self.var.get, self.var.set

    def check(self, *args):
        if self.get().isdigit():
            # the current value is only digits; allow this
            val = int(self.get())
            if(val>=0 and val<100):
                self.old_value = self.get()
            else:
                self.old_value = ''
                self.set(self.old_value)
        else:
            # there's non-digit characters in the input; reject this
            self.old_value =''
            self.set(self.old_value)


root = Tk()
root.title("FIFA_APP")
machineLearningObj = TrainTestTkkApp()
listsFromFile = machineLearningObj.returnOptions()

e = Entry(root, width=100, borderwidth = 5,state = 'readonly')
e.grid(row=0, column=0, columnspan = 5, padx=10, pady=10)

def show_price():
    data = [[entry_1.get(),entry_2.get(),entry_3.get(),entry_4.get(),entry_5.get(),entry_6.get(),entry_7.get(),entry_8.get(),returnPositionOfObjectInArray(varPos.get(),listsFromFile[0]),returnPositionOfObjectInArray(varCountry.get(),listsFromFile[1]),returnPositionOfObjectInArray(varClub.get(),listsFromFile[2]),returnPositionOfObjectInArray(varWork.get(),listsFromFile[3])]]
    model_opt = clicked.get()
    model_opt = func(model_opt)
    result = machineLearningObj.checkNewFifaPlayer(data,model_opt)
    e.config(state='normal')
    e.delete(0, END)
    string = "Oczekiwana wartosc pilkarza to: {:.0f} ".format(float(result))
    e.insert(0, string)
    e.config(state='readonly')


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
entry_1 = Lotfi(root, width=10, borderwidth = 5,)
entry_2 = Lotfi(root, width=10, borderwidth = 5,)
entry_3 = Lotfi(root, width=10, borderwidth = 5,)
entry_4 = Lotfi(root, width=10, borderwidth = 5,)
entry_5 = Lotfi(root, width=10, borderwidth = 5,)
entry_6 = Lotfi(root, width=10, borderwidth = 5,)
entry_7 = Lotfi(root, width=10, borderwidth = 5,)
entry_8 = Lotfi(root, width=10, borderwidth = 5,)
l1 = Label(root, text ='WeakFoot')
l2 = Label(root, text ='SkillsMoves')
l3 = Label(root, text ='Pace')
l4 = Label(root, text ='Shooting')
l5 = Label(root, text ='Passing')
l6 = Label(root, text ='Dribbling')
l7 = Label(root, text ='Defending')
l8 = Label(root, text ='Phyiscality')
l9 = Label(root, text ='Position')
l10 = Label(root, text ='Country')
l11= Label(root, text ='Club')
l12 = Label(root, text ='WorkRate')
l1.grid(row=3,column=0)
l2.grid(row=3,column=1)
l3.grid(row=3,column=2)
l4.grid(row=3,column=3)
l5.grid(row=3,column=4)
l6.grid(row=5,column=0)
l7.grid(row=5,column=1)
l8.grid(row=5,column=2)
l9.grid(row=5,column=3)
l10.grid(row=5,column=4)
l11.grid(row=7,column=0)
l12.grid(row=7,column=1)
entry_1.grid(row=4, column=0, columnspan = 1, padx=10, pady=10)
entry_2.grid(row=4, column=1, columnspan = 1, padx=10, pady=10)
entry_3.grid(row=4, column=2, columnspan = 1, padx=10, pady=10)
entry_4.grid(row=4, column=3, columnspan = 1, padx=10, pady=10)
entry_5.grid(row=4, column=4, columnspan = 1, padx=10, pady=10)
entry_6.grid(row=6, column=0, columnspan = 1, padx=10, pady=10)
entry_7.grid(row=6, column=1, columnspan = 1, padx=10, pady=10)
entry_8.grid(row=6, column=2, columnspan = 1, padx=10, pady=10)
btn_1.grid(row=1,column=0)
btn_2.grid(row=1,column=1)
btn_3.grid(row=1,column=2)
btn_4.grid(row=1,column=3)
btn_5.grid(row=1,column=4)
getPrice.grid(row=8, column = 3)


def func(var):
    if(var=='Linear Regression'):
        return 1
    elif(var=='Gradient Boosting'):
        return 2
    elif(var=='Extra Trees'):
        return 3
    elif(var=='Random Forest'):
        return 4
    elif(var=='Decision Tree'):
        return 5

model_options =[
    'Linear Regression',
    'Gradient Boosting',
    'Extra Trees',
    'Random Forest',
    'Decision Tree'
    ]
def returnPositionOfObjectInArray(number,array):
    index = array.index(number)
    return index


varClub = StringVar()
varCountry = StringVar()
varPos = StringVar()
varWork = StringVar()

dropCountries = Combobox(root,textvariable=varCountry , values=listsFromFile[1])
dropCountries.grid(row=6, column=4,sticky = 'ew')
dropClubs = Combobox(root,textvariable=varClub , values=listsFromFile[2])
dropClubs.grid(row=8, column=0,sticky = 'ew')
workRate = Combobox(root,textvariable=varWork , values=listsFromFile[3])
workRate.grid(row=8, column=1,sticky = 'ew')
position = Combobox(root,textvariable=varPos , values=listsFromFile[0])
position.grid(row=6, column=3,sticky = 'ew')
clicked = StringVar()
clicked.set('Linear Regression')
drop = OptionMenu(root,clicked,*model_options)
drop.grid(row=8, column=2,sticky = 'ew')
root.mainloop()