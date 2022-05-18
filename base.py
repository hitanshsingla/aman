#!/usr/bin/env python
# coding: utf-8

from random import randint
from forms import UserInfoForm
from flask import Flask, render_template, request, url_for, redirect
import algo


protein = ['Yogurt(1 cup)', 'Cooked meat(85g)', 'Cooked fish(100g)', '1 whole egg + 4 egg whites', 'Tofu(125g)']
fruit = ['Berries(80g)', 'Apple', 'Orange', 'Banana', 'Dried Fruit(Handfull)', 'Fruit Juice(125ml)']
vegetable = ['Any vegetable(80g)', 'Leafy greens(Any Amount)']
grains = ['Cooked Grain(150g)', 'Whole Grain Bread(1 slice)', 'Half Large Potato(75g)', 'Oats(250g)', '2 corn tortillas']
protein_snack = ['Soy nuts(30g)', 'Low fat milk(250ml)', 'Hummus(4 Tbsp)', 'Cottage cheese (125g)',
                 'Flavored yogurt(125g)']
taste_enhancer = ['2 TSP (10 ml) olive oil', '2 TBSP (30g) reduced-calorie salad dressin', '1/4 medium avocado',
                  'Small handful of nuts', '1/2 ounce  grated Parmesan cheese', '1 TBSP (20g) jam, jelly, honey, syrup, sugar']


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


def bfcalc(tdee):
    breakfast = protein[randint(0, len(protein) - 1)] + ", "
    breakfast += fruit[randint(0, len(fruit) - 1)]

    if tdee >= 2200:
        breakfast += ", " + grains[randint(0, len(grains) - 1)]

    return breakfast


def s1calc(tdee):
    snack1 = ""
    if tdee >= 1800:
        snack1 = protein_snack[randint(0, len(protein_snack) - 1)]

    return snack1


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


def s2calc(tdee):
    snack2 = protein_snack[randint(0, len(protein_snack) - 1)] + ", "
    snack2 += vegetable[randint(0, len(vegetable) - 1)]
    return snack2


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

            tdee = algo.calc_tdee(name, weight, height, age, gender, phys_act)
            return redirect(url_for('result', tdee=tdee))

    return render_template('home.html', title="Diet App", form=form)


@app.route('/result', methods=['GET', 'POST'])
def result():
    tdee = request.args.get('tdee')
    if tdee is None:
        return render_template('error.html', title="Error Page")

    tdee = float(tdee)
    breakfast = algo.bfcalc(tdee)
    snack1 = algo.s1calc(tdee)
    lunch = algo.lcalc(tdee)
    snack2 = algo.s2calc(tdee)
    dinner = algo.dcalc(tdee)
    snack3 = algo.s3calc(tdee)
    return render_template('result.html', title="Result", breakfast=breakfast, snack1=snack1, lunch=lunch,
                           snack2=snack2, dinner=dinner, snack3=snack3)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

# In[ ]:
