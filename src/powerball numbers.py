# Libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
file = 'Data/powerball26-05.html'
with open(file, 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')

#Parsing for balls
white_ball_divs = soup.find_all('div', class_= 'form-control col white-balls item-powerball')
red_ball_divs = soup.find_all('div', class_ = 'form-control col powerball item-powerball')

white_balls = []
red_balls = []
for div in white_ball_divs:
    raw_num = div.text.strip()

    if raw_num.isdigit():
        white_balls.append(int(raw_num))

for div in red_ball_divs:
    raw_num = div.text.strip()

    if raw_num.isdigit():
        red_balls.append(int(raw_num))

# Parsing for date splits
dates = soup.find_all('h5', class_ = 'card-title')
w_day = []
day = []
month_str = []
year = []

for date in dates:
    raw_date = date.text.strip()
    year.append(int(raw_date[12:17]))
    w_day.append(raw_date[0:3])
    month_str.append(raw_date[5:8])
    day.append(int(raw_date[9:11].strip(',')))

month_int = []

for month in month_str:
    if month == 'Jan':
        month_int.append(1)
    elif month == 'Feb':
        month_int.append(2)
    elif month == 'Mar':
        month_int.append(3)
    elif month == 'Apr':
        month_int.append(4)
    elif month == 'May':
        month_int.append(5)
    elif month == 'Jun':
        month_int.append(6)
    elif month == 'Jul':
        month_int.append(7)
    elif month == 'Aug':
        month_int.append(8)
    elif month == 'Sep':
        month_int.append(9)
    elif month == 'Oct':
        month_int.append(10)
    elif month == 'Nov':
        month_int.append(11)
    else:
        month_int.append(12)

som_eom = []

for x in day:
    if x <= 15.5:
        som_eom.append('Start of Month')
    else:
        som_eom.append('End of Month')

d = {
    'week_day' : w_day,
    'day' : day,
    'month_str' : month_str,
    'month_int' : month_int,
    'year' : year,

    'start_end_month' : som_eom,

    'white_ball1': white_balls[0::5],
    'white_ball2': white_balls[1::5],
    'white_ball3': white_balls[2::5],
    'white_ball4': white_balls[3::5],
    'white_ball5': white_balls[4::5],
    'red_ball': red_balls
}
numbers2605 = pd.DataFrame(data = d)

numbers2605['date_drawn'] = pd.to_datetime(numbers2605[['day', 'month_int', 'year']].rename(columns = {'month_int' : 'month'}))
numbers2605['sum_white'] = numbers2605['white_ball1'] + numbers2605['white_ball2'] + numbers2605['white_ball3'] + numbers2605['white_ball4'] + numbers2605['white_ball5']
numbers2605['sum_white_red'] = numbers2605['white_ball1'] + numbers2605['white_ball2'] + numbers2605['white_ball3'] + numbers2605['white_ball4'] + numbers2605['white_ball5'] + numbers2605['red_ball']

# Sum balls high/low
high_low_w = []
for x in numbers2605['sum_white']:
    if x <= 172.5:
        high_low_w.append('Low')
    else:
        high_low_w.append('High')
high_low_wr = []
for x in numbers2605['sum_white_red']:
    if x <= 185.5:
        high_low_wr.append('Low')
    else:
        high_low_wr.append('High')
numbers2605['sum_white_hl'] = high_low_w
numbers2605['sum_white_red_hl'] = high_low_wr

# Individual balls high/low
wb1_hl = []
wb2_hl = []
wb3_hl = []
wb4_hl = []
wb5_hl = []
rb_hl = []
for x in numbers2605['white_ball1']:
    if x <= 34.5:
        wb1_hl.append('Low')
    else:
        wb1_hl.append('High')
for x in numbers2605['white_ball2']:
    if x <= 34.5:
        wb2_hl.append('Low')
    else:
        wb2_hl.append('High')
for x in numbers2605['white_ball3']:
    if x <= 34.5:
        wb3_hl.append('Low')
    else:
        wb3_hl.append('High')
for x in numbers2605['white_ball4']:
    if x <= 34.5:
        wb4_hl.append('Low')
    else:
        wb4_hl.append('High')
for x in numbers2605['white_ball5']:
    if x <= 34.5:
        wb5_hl.append('Low')
    else:
        wb5_hl.append('High')
for x in numbers2605['red_ball']:
    if x <= 13:
        rb_hl.append('Low')
    else:
        rb_hl.append('High')
numbers2605['white_ball1_high_low'] = wb1_hl
numbers2605['white_ball2_high_low'] = wb2_hl
numbers2605['white_ball3_high_low'] = wb3_hl
numbers2605['white_ball4_high_low'] = wb4_hl
numbers2605['white_ball5_high_low'] = wb5_hl
numbers2605['red_ball_high_low'] = rb_hl


# individual balls odd/even
wb1_oe = []
wb2_oe = []
wb3_oe = []
wb4_oe = []
wb5_oe = []
for x in numbers2605['white_ball1']:
    if x % 2 == 0:
        wb1_oe.append('Even')
    else:
        wb1_oe.append('Odd')
for x in numbers2605['white_ball2']:
    if x % 2 == 0:
        wb2_oe.append('Even')
    else:
        wb2_oe.append('Odd')
for x in numbers2605['white_ball3']:
    if x % 2 == 0:
        wb3_oe.append('Even')
    else:
        wb3_oe.append('Odd')
for x in numbers2605['white_ball4']:
    if x % 2 == 0:
        wb4_oe.append('Even')
    else:
        wb4_oe.append('Odd')
for x in numbers2605['white_ball5']:
    if x % 2 == 0:
        wb5_oe.append('Even')
    else:
        wb5_oe.append('Odd')
numbers2605['white_ball1_odd_even'] = wb1_oe
numbers2605['white_ball2_odd_even'] = wb2_oe
numbers2605['white_ball3_odd_even'] = wb3_oe
numbers2605['white_ball4_odd_even'] = wb4_oe
numbers2605['white_ball5_odd_even'] = wb5_oe

# Sum balls odd/even
wb_sums = []
wrb_sums = []

for x in numbers2605['sum_white']:
    if x % 2 == 0:
        wb_sums.append('Even')
    else:
        wb_sums.append('Odd')

for x in numbers2605['sum_white_red']:
    if x % 2 == 0:
        wrb_sums.append('Even')
    else:
        wrb_sums.append('Odd')

numbers2605['white_sums_odd_even'] = wb_sums
numbers2605['white_red_sums_odd_even'] = wrb_sums

# Sorted data frame
numbers2605 = numbers2605[[
    'week_day',
    'day',
    'month_str',
    'month_int',
    'year',
    'date_drawn',
    'start_end_month',
    'white_ball1',
    'white_ball1_high_low',
    'white_ball1_odd_even',
    'white_ball2',
    'white_ball2_high_low',
    'white_ball2_odd_even',
    'white_ball3',
    'white_ball3_high_low',
    'white_ball3_odd_even',
    'white_ball4',
    'white_ball4_high_low',
    'white_ball4_odd_even',
    'white_ball5',
    'white_ball5_high_low',
    'white_ball5_odd_even',
    'sum_white',
    'sum_white_hl',
    'white_sums_odd_even',
    'red_ball',
    'red_ball_high_low',
    'sum_white_red',
    'sum_white_red_hl',
    'white_red_sums_odd_even'

]]

numbers2605 = numbers2605.sort_values(by = 'date_drawn', ascending = True).reset_index(drop = True)

numbers2605_melted = numbers2605.melt(
    id_vars = ['date_drawn'], 
    value_vars = ['white_ball1', 'white_ball2', 'white_ball3', 'white_ball4', 'white_ball5'], 
    value_name = 'drawn_number'
)

pre2015 = numbers2605_melted[numbers2605_melted['date_drawn'] < '2015-10-07']
post2015 = numbers2605_melted[numbers2605_melted['date_drawn'] >= '2015-10-07']

draws_post2015 = numbers2605[numbers2605['date_drawn'] >= '2015-10-07']
draws_post2015 = draws_post2015.reset_index(drop = True)


white_lagged_cols = ['sum_white', 'sum_white_hl', 'white_sums_odd_even']

for col in white_lagged_cols:
    draws_post2015[f'{col}_lag1'] = draws_post2015[col].shift(1)

white_model_df = draws_post2015.dropna( subset = [f'{col}_lag1' for col in white_lagged_cols])
white_model_df = white_model_df.reset_index(drop = True)

white_cat_cols = ['week_day', 'start_end_month', 'sum_white_hl_lag1', 'white_sums_odd_even_lag1']
white_num_cols = ['day', 'month_int', 'year', 'sum_white_lag1']

preprocessor = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), white_cat_cols), ('num', 'passthrough', white_num_cols)])

white_x_pro = preprocessor.fit_transform(white_model_df)
feat_names = preprocessor.get_feature_names_out()
white_x = pd.DataFrame(white_x_pro, columns = feat_names)

white_model_df['winning_numbers'] = white_model_df[['white_ball1', 'white_ball2', 'white_ball3', 'white_ball4', 'white_ball5']].values.tolist()
mlb = MultiLabelBinarizer(classes = range(1, 70))
white_y = pd.DataFrame(mlb.fit_transform(white_model_df['winning_numbers']), columns = mlb.classes_)


white_xgc = XGBClassifier(random_state = 42, device = 'cuda')

white_x_train, white_x_test, white_y_train, white_y_test = train_test_split(white_x, white_y, test_size = 0.2, shuffle = False)

white_xgc.fit(white_x_train, white_y_train)
y_pred = white_xgc.predict_proba(white_x_test)

top5 = np.argsort(y_pred, axis = 1)[:, -5:][:, ::-1]

white_pdd = top5 + 1

act_draws = mlb.inverse_transform(white_y_test.values)

match_hist = []

for x in range(len(white_pdd)):
    pred_set = set(white_pdd[x])
    act_set = set(act_draws[x])
    correct = len(pred_set.intersection(act_set))
    match_hist.append(correct)

match_sum = Counter(match_hist)
print('--- Model Evaluation on White Ball Test Data ---')
for x in range(6):
    count = match_sum.get(x, 0)
    print(f'matched {x} balls: {count} times')

red_lagged_cols = ['red_ball_high_low']

for col in red_lagged_cols:
    draws_post2015[f'{col}_lag1'] = draws_post2015[col].shift(1)

red_model_df = draws_post2015.dropna( subset = [f'{col}_lag1' for col in red_lagged_cols])
red_model_df = red_model_df.reset_index(drop = True)

red_cat_cols = ['week_day', 'start_end_month', 'red_ball_high_low_lag1']
red_num_cols = ['day', 'month_int', 'year']

preprocessor = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), red_cat_cols), ('num', 'passthrough', red_num_cols)])

red_x_pro = preprocessor.fit_transform(red_model_df)
feat_names = preprocessor.get_feature_names_out()
red_x = pd.DataFrame(red_x_pro, columns = feat_names)

red_model_df['winning_numbers'] = red_model_df[['red_ball']].values.tolist()
mlb = MultiLabelBinarizer(classes = range(1, 27))
red_y = pd.DataFrame(mlb.fit_transform(red_model_df['winning_numbers']), columns = mlb.classes_)


red_xgc = XGBClassifier(random_state = 42, device = 'cuda')

red_x_train, red_x_test, red_y_train, red_y_test = train_test_split(red_x, red_y, test_size = 0.2, shuffle = False)

red_xgc.fit(red_x_train, red_y_train)
y_pred = red_xgc.predict_proba(red_x_test)

top5 = np.argsort(y_pred, axis = 1)[:, -1:]

red_pdd = top5 + 1

act_draws = mlb.inverse_transform(red_y_test.values)

match_hist = []

for x in range(len(red_pdd)):
    pred_set = set(red_pdd[x])
    act_set = set(act_draws[x])
    correct = len(pred_set.intersection(act_set))
    match_hist.append(correct)

match_sum = Counter(match_hist)
print('\n--- Model Evaluation on Red Ball Test Data ---')
for x in range(2):
    count = match_sum.get(x, 0)
    print(f'matched {x} balls: {count} times')

print('\n--- XGBoost Simulated Powerball Tickets (First 5 Draws) ---')

for x in range(5):
    white_ticket = ', '.join(map(str, white_pdd[x]))

    red_ticket = str(red_pdd[x][0])
    print(f'Draw {x + 1}: White Balls [{white_ticket}] | Powerball [{red_ticket}]')



# Old Rules Vs New Rules
fig, axes = plt.subplots(1, 2, figsize = (15, 6), sharey = True)

sns.histplot(data = pre2015, x = 'drawn_number', ax = axes[0], bins = 59, discrete = True)
axes[0].set_title('Old Rules (2005 - Oct 2015) 59 ball pool')
axes[0].set_xlabel('White Ball Number')
axes[0].set_ylabel('Total Times Drawn')

sns.histplot(data = post2015, x = 'drawn_number', ax = axes[1], bins = 69, discrete = True)
axes[1].set_title('New Rules (Oct 2015 - 2026) 69 ball pool')
axes[1].set_xlabel('White Ball Number')
axes[1].set_ylabel('Total Times Drawn')

plt.tight_layout()
plt.show()
