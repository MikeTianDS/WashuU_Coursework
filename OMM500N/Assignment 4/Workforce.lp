\* Workforce Planning *\
Minimize
total_salary: - 3000 l_1 - 3000 l_2 - 3000 l_3 - 3000 l_4 - 3000 l_5
 - 3000 l_6 + 3300 tr_1 + 3300 tr_2 + 3300 tr_3 + 3300 tr_4 + 3300 tr_5
 + 3300 tr_6 + 3000 u_1 + 3000 u_2 + 3000 u_3 + 3000 u_4 + 3000 u_5 + 3000 u_6
Subject To
initial_trained: tr_1 = 0
initial_untrained: u_1 = 145
layoff_limit_1: l_1 - u_1 <= 0
layoff_limit_2: l_2 - u_2 <= 0
layoff_limit_3: l_3 - u_3 <= 0
layoff_limit_4: l_4 - u_4 <= 0
layoff_limit_5: l_5 - u_5 <= 0
layoff_limit_6: l_6 - u_6 <= 0
satisfy_demand_1: - l_1 + 1.2 tr_1 + u_1 - x_1 >= 100
satisfy_demand_2: - l_2 + 1.2 tr_2 + u_2 - x_2 >= 100
satisfy_demand_3: - l_3 + 1.2 tr_3 + u_3 - x_3 >= 115
satisfy_demand_4: - l_4 + 1.2 tr_4 + u_4 - x_4 >= 125
satisfy_demand_5: - l_5 + 1.2 tr_5 + u_5 - x_5 >= 140
satisfy_demand_6: - l_6 + 1.2 tr_6 + u_6 - x_6 >= 150
trained_balance_1: tr_1 - tr_2 + 0.9 x_1 = 0
trained_balance_2: tr_2 - tr_3 + 0.9 x_2 = 0
trained_balance_3: tr_3 - tr_4 + 0.9 x_3 = 0
trained_balance_4: tr_4 - tr_5 + 0.9 x_4 = 0
trained_balance_5: tr_5 - tr_6 + 0.9 x_5 = 0
training_limit_1: l_1 - u_1 + x_1 <= 0
training_limit_2: l_2 - u_2 + x_2 <= 0
training_limit_3: l_3 - u_3 + x_3 <= 0
training_limit_4: l_4 - u_4 + x_4 <= 0
training_limit_5: l_5 - u_5 + x_5 <= 0
training_limit_6: l_6 - u_6 + x_6 <= 0
untrained_balance_1: - 0.95 l_1 + 0.95 u_1 - u_2 - 0.85 x_1 = 0
untrained_balance_2: - 0.95 l_2 + 0.95 u_2 - u_3 - 0.85 x_2 = 0
untrained_balance_3: - 0.95 l_3 + 0.95 u_3 - u_4 - 0.85 x_3 = 0
untrained_balance_4: - 0.95 l_4 + 0.95 u_4 - u_5 - 0.85 x_4 = 0
untrained_balance_5: - 0.95 l_5 + 0.95 u_5 - u_6 - 0.85 x_5 = 0
Bounds
0 <= l_1
0 <= l_2
0 <= l_3
0 <= l_4
0 <= l_5
0 <= l_6
0 <= tr_1
0 <= tr_2
0 <= tr_3
0 <= tr_4
0 <= tr_5
0 <= tr_6
0 <= u_1
0 <= u_2
0 <= u_3
0 <= u_4
0 <= u_5
0 <= u_6
0 <= x_1
0 <= x_2
0 <= x_3
0 <= x_4
0 <= x_5
0 <= x_6
End
