#!/usr/bin/env python
# coding: utf-8

from fileinput import filename
from random import randint
from forms import UserInfoForm
from flask import Flask, render_template, request, url_for, redirect
from algo_custom import Weight_Loss, Weight_Gain
import os

file_name = 'mini_db.csv'
if os.path.exists(file_name):
    f = open(file_name, 'a')
else:
    f = open(file_name, 'w')
    f.write('name,weight,height,age,gender,pa')
    f.flush()

protein = ['Yogurt(1 cup)', 'Cooked meat(85g)', 'Cooked fish(100g)', '1 whole egg + 4 egg whites', 'Tofu(125g)']
fruit = ['Berries(80g)', 'Apple', 'Orange', 'Banana', 'Dried Fruit(Handfull)', 'Fruit Juice(125ml)']
vegetable = ['Any vegetable(80g)', 'Leafy greens(Any Amount)']
grains = ['Cooked Grain(150g)', 'Whole Grain Bread(1 slice)', 'Half Large Potato(75g)', 'Oats(250g)',
          '2 corn tortillas']
protein_snack = ['Soy nuts(30g)', 'Low fat milk(250ml)', 'Hummus(4 Tbsp)', 'Cottage cheese (125g)',
                 'Flavored yogurt(125g)']
taste_enhancer = ['2 TSP (10 ml) olive oil', '2 TBSP (30g) reduced-calorie salad dressin', '1/4 medium avocado',
                  'Small handful of nuts', '1/2 ounce  grated Parmesan cheese',
                  '1 TBSP (20g) jam, jelly, honey, syrup, sugar']


# calculates total daily energy expenditure
def calc_tdee(name, weight, height, age, gender, phys_act):
    if gender == 'Female':
        bmr = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        bmr = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)

    if phys_act == 'value1':
        tdee = bmr * 1.2
    elif phys_act == 'value2':
        tdee = bmr * 1.375
    elif phys_act == 'value3':
        tdee = bmr * 1.55
    elif phys_act == 'value4':
        tdee = bmr * 1.735
    else:
        tdee = bmr * 1.9
    return tdee


# based on t-dee, it calculates breakfast
def bfcalc(tdee):
    breakfast = protein[randint(0, len(protein) - 1)] + ", "
    breakfast += fruit[randint(0, len(fruit) - 1)]

    if tdee >= 2200:
        breakfast += ", " + grains[randint(0, len(grains) - 1)]

    return breakfast


# snack 1 calculator
def s1calc(tdee):
    snack1 = ""
    if tdee >= 1800:
        snack1 = protein_snack[randint(0, len(protein_snack) - 1)]

    return snack1


# lunch
def lcalc(tdee):
    lunch = ""
    lunch += protein[randint(0, len(protein) - 1)] + ", "
    lunch += vegetable[randint(0, len(vegetable) - 1)] + ", "
    lunch += "Leafy greens, "
    lunch += taste_enhancer[randint(0, len(taste_enhancer) - 1)] + ", "
    lunch += grains[randint(0, len(grains) - 1)]

    if (tdee >= 1500):
        lunch += ", " + fruit[randint(0, len(fruit) - 1)]

    if (tdee >= 1800):
        lunch += ", " + protein[randint(0, len(protein) - 1)] + ", "
        lunch += vegetable[randint(0, len(vegetable) - 1)]
    return lunch


# snack 2
def s2calc(tdee):
    snack2 = protein_snack[randint(0, len(protein_snack) - 1)] + ", "
    snack2 += vegetable[randint(0, len(vegetable) - 1)]
    return snack2


# dinner
def dcalc(tdee):
    dinner = ""
    dinner += protein[randint(0, len(protein) - 1)] + ", "
    dinner += "2 vegetables 80g, "
    dinner += "Leafy Greens, "
    dinner += grains[randint(0, len(grains) - 1)] + ", "
    dinner += taste_enhancer[randint(0, len(taste_enhancer) - 1)]

    if tdee >= 1500:
        dinner += ", " + protein[randint(0, len(protein) - 1)]

    if tdee >= 2200:
        dinner += ", " + grains[randint(0, len(grains) - 1)] + ", "
        dinner += taste_enhancer[randint(0, len(taste_enhancer) - 1)]
    return dinner


# snack 3
def s3calc(tdee):
    snack3 = fruit[randint(0, len(fruit) - 1)]
    return snack3


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '945a61eeffcee883e3b261a47b31ae47'


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UserInfoForm()
    if form.validate_on_submit():
        if request.method == 'POST':
            name = request.form['name']
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            age = int(request.form['age'])
            gender = request.form['gender']
            phys_act = request.form['physical_activity']
            vnv = request.form['food_pref']

            return redirect(url_for('result',
                                    name=name,
                                    weight=weight,
                                    height=height,
                                    age=age,
                                    gender=gender,
                                    phys_act=phys_act,
                                    vnv=vnv))

    return render_template('home.html', title="Diet App", form=form)


@app.route('/result', methods=['GET', 'POST'])
def result():
    name = request.args.get('name')
    weight = float(request.args.get('weight'))
    height = float(request.args.get('height'))
    age = int(request.args.get('age'))
    gender = request.args.get('gender')
    phys_act = request.args.get('phys_act')
    vnv = request.args.get('vnv')

    f.write(f"\n{name},{weight},{age},{gender},{phys_act}")
    f.flush()

    weight_loss_diet = Weight_Loss(weight, height, age, vnv)
    weight_gain_diet = Weight_Gain(weight, height, age, vnv)

    tdee = calc_tdee(name, weight, height, age, gender, phys_act)
    if tdee is None:
        return render_template('error.html', title="Error Page")

    tdee = float(tdee)
    breakfast = bfcalc(tdee)
    snack1 = s1calc(tdee)
    lunch = lcalc(tdee)
    snack2 = s2calc(tdee)
    dinner = dcalc(tdee)
    snack3 = s3calc(tdee)
    return render_template('result.html', title="Result", name=name, breakfast=breakfast, snack1=snack1, lunch=lunch,
                           snack2=snack2, dinner=dinner, snack3=snack3, weight_loss_diet=weight_loss_diet,
                           weight_gain_diet=weight_gain_diet)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

