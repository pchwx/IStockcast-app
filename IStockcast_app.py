import streamlit as st 
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import random
import math
import matplotlib.patches as mpatches 
import io
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")

image = Image.open('Logo3.PNG')
st.image(image,width=400)
global start_button, upper, lower, init_temp, stop, n_test, test_data, train_data

def adf_test(series):
    result=adfuller(series)
    return result[1]

def check_stationary(df, lower, upper): # รับ lower and upper ด้วย
        dl = 0
        pvl = adf_test(df[lower])
        if pvl > 0.05:
            for i in range(3):
                dl += 1
                temp = df[lower] - df[lower].shift(dl)
                pvl = adf_test(temp.dropna())
                if pvl <= 0.05:
                    break
        print("\nFinal d lower = ",dl)

        du = 0
        pvu = adf_test(df[upper])
        if pvu > 0.05:
            for i in range(3):
                du += 1
                temp = df[upper] - df[upper].shift(du)
                pvu = adf_test(temp.dropna())
                if pvu <= 0.05:
                    break
        print("\nFinal d upper = ",du)
        return dl,du
#---------------------------------------------------------------------------#
#   Simulated Annealing

# Define the initial state
def initial_state():
    index0 = random.randint(0, 1)
    index1 = random.randint(0, 1)
    index2_5 = [random.randint(0, 5) for _ in range(4)]
    initial_state = [index0] + [index1] + index2_5
    return initial_state

# Define the neighbor function
def neighbor_state(current_state):
    neighbor_state = current_state.copy()
    index = [0,1,2,3,4,5]

    #rand = random.randint(1,len(neighbor_state))
    rand = random.randint(1,4)

    for i in range(rand):

        index_to_change = random.choice(index)

        if index_to_change == 0 or index_to_change == 1:
            if neighbor_state[index_to_change] == 0:
                neighbor_state[index_to_change] = 1
            elif neighbor_state[index_to_change] == 1:
                neighbor_state[index_to_change] = 0

        else:     
            neighbor_state[index_to_change] = random.choice([i for i in list(range(6)) if i not in [neighbor_state[index_to_change]]])

        index = [i for i in index if i not in [index_to_change]]

    return neighbor_state

# Define the objective function to minimize
def MDE2(actual_low_value, forecast_low_value, actual_high_value, forecast_high_value, test_data):
            MDE2 = (sum(((actual_low_value-forecast_low_value)**2 + (actual_high_value-forecast_high_value)**2)/2))/test_data
            return MDE2

def objective_function(state,df,dl,du,train_data,test_data, lower, upper): # รับ lower and upper ด้วย
    is_const_low = state[0]
    is_const_up = state[1]
    p_low = state[2]
    q_low = state[3]
    p_up = state[4]
    q_up = state[5]
    M = 100

    #Create ARIMA Model 

    if is_const_low == 0:
        Lower_Model_nc = sm.tsa.statespace.SARIMAX(df[lower][:train_data], trend='n', order=(p_low,dl,q_low))
        lower_model_fit = Lower_Model_nc.fit(disp=False)
    else:
        Lower_Model_c = sm.tsa.statespace.SARIMAX(df[lower][:train_data], trend='c', order=(p_low,dl,q_low))
        lower_model_fit = Lower_Model_c.fit(disp=False)
    
    if is_const_up == 0:
        Upper_Model_nc = sm.tsa.statespace.SARIMAX(df[upper][:train_data], trend='n', order=(p_up,du,q_up))
        Upper_Model_fit = Upper_Model_nc.fit(disp=False)
    else:
        Upper_Model_c = sm.tsa.statespace.SARIMAX(df[upper][:train_data], trend='c', order=(p_up,du,q_up))
        Upper_Model_fit = Upper_Model_c.fit(disp=False)

    pred_lower = lower_model_fit.forecast(test_data)
    pred_upper = Upper_Model_fit.forecast(test_data)

    #find error MDE^2
    Error_MDE2 = MDE2(df[lower][train_data:], pred_lower, df[upper][train_data:], pred_upper ,test_data)

    e = 0
    for i in pred_upper.index:
        if pred_upper[i] < pred_lower[i]:
            e += 1

    obj = Error_MDE2 + M*e

    return obj

# Define the acceptance probability function
def acceptance_probability(current_energy, new_energy, temperature):
    if new_energy < current_energy:
        return 1.0
    return math.exp((current_energy - new_energy) / temperature)

# Define the acceptance probability function
def acceptance_probability(current_energy, new_energy, temperature):
    if new_energy < current_energy:
        return 1.0
    return math.exp((current_energy - new_energy) / temperature)

def SA(df, dl, du, init_temp, train_data, test_data, numstop, lower, upper, progess_box):
    # Initialize the temperature and current state
    temperature = init_temp
    t0 = temperature

    current_state = initial_state()
    current_energy = objective_function(current_state,df,dl,du,train_data,test_data, lower, upper)
    best_state = current_state
    best_energy = current_energy
    count = 0
    count_stop = 0
    text = ""

    # Main loop
    round_start_time = time.time() 
    while temperature > 0.1:
        count += 1
        
        count_stop += 1

        # Generate a new candidate solution
        new_state = neighbor_state(current_state)

        # Calculate the energies
        new_energy = objective_function(new_state,df,dl,du,train_data,test_data, lower, upper) # รับ lower and upper ด้วย
        

        # Decide whether to accept the new state
        
        if acceptance_probability(current_energy, new_energy, temperature) > random.random():
            current_state = new_state
            current_energy = new_energy

        # Update the best state if necessary
        if current_energy < best_energy :
        #if objective_function(current_state) < objective_function(best_state):
            best_state = current_state
            best_energy = current_energy
            count_stop = 0
            if st.session_state['progress_bar_value'] <= 95:
                st.session_state['progress_bar_value'] += np.random.randint(2,6)
                my_bar.progress(text=f"Progress {st.session_state['progress_bar_value']}%",value=st.session_state['progress_bar_value'])

        round_end_time = time.time()  # Record end time for the round
        round_time = round_end_time - round_start_time
        print("Round", count,"\thas error:", round(best_energy,4), \
            "\tmodel:", best_state, "\tTime:", round(round_time,4), "\tNow at", current_state,"\tstop:", count_stop, "Temp:", round(temperature,4))
        
        text += f"Round {count} \thas error: {round(best_energy,4)} \
            \tmodel: {best_state} \tTime: {round(round_time,4)} \tNow at {current_state} \tstop: {count_stop} \
            Temp: {round(temperature,4)}\n"
        #progess_box.markdown(f"```\n{text}\n```") 
        if count_stop >= numstop:
            break

        temperature = t0/np.log(count+1)
        if st.session_state['progress_bar_value'] <= 95 and (count%100==0):
            st.session_state['progress_bar_value'] += np.random.randint(2,6)
            my_bar.progress(text=f"Progress {st.session_state['progress_bar_value']}%",value=st.session_state['progress_bar_value'])
        

    return best_state   
    
#---------------------------------------------------------------------------#
#start
def start_it():
    global start_button, upper, lower, init_temp, stop, n_test, test_data, train_data
    #global start_button, upper, lower, init_temp, stop, n_test, test_data, train_data, result_table, dfpred

    st.session_state['progress_bar_value'] = 0
    my_bar.progress(text=f"Progress {st.session_state['progress_bar_value']}%",value=st.session_state['progress_bar_value'])

    dl, du = check_stationary(df, lower, upper)
    print(df, dl, du, init_temp, train_data, test_data, stop, lower, upper, progess_box)
    best_solution = SA(df, dl, du, init_temp, train_data, test_data, stop, lower, upper, progess_box) 

    st.session_state['progress_bar_value'] = 100
    my_bar.progress(text="Finish!",value=st.session_state['progress_bar_value'])

    print("Best solution found:", best_solution)
    print("Objective value:", round(objective_function(best_solution,df,dl,du,train_data,test_data, lower, upper),4)) 
    text = f"Best solution found: {best_solution}"
    #progess_box.markdown(f"```\n{text}\n```")
    text = f"Objective value: {round(objective_function(best_solution,df,dl,du,train_data,test_data, lower, upper),4)}"
    #progess_box.markdown(f"```\n{text}\n```") 
    
    state = best_solution 

    is_const_low = state[0]
    is_const_up = state[1]
    p_low = state[2]
    q_low = state[3]
    p_up = state[4]
    q_up = state[5]
        

    #Create ARIMA Model 
    if is_const_low == 0:
        Lower_Model_nc = sm.tsa.statespace.SARIMAX(df['Low'][:train_data], trend='n', order=(p_low,dl,q_low))
        lower_model_fit = Lower_Model_nc.fit(disp=False)
    else:
        Lower_Model_c = sm.tsa.statespace.SARIMAX(df['Low'][:train_data], trend='c', order=(p_low,dl,q_low))
        lower_model_fit = Lower_Model_c.fit(disp=False)

    if is_const_up == 0:
        Upper_Model_nc = sm.tsa.statespace.SARIMAX(df['High'][:train_data], trend='n', order=(p_up,du,q_up))
        Upper_Model_fit = Upper_Model_nc.fit(disp=False)
    else:
        Upper_Model_c = sm.tsa.statespace.SARIMAX(df['High'][:train_data], trend='c', order=(p_up,du,q_up))
        Upper_Model_fit = Upper_Model_c.fit(disp=False)

    pred_lower = lower_model_fit.forecast(test_data+5)
    pred_upper = Upper_Model_fit.forecast(test_data+5)

    dfact_pred = pd.DataFrame(df[lower][train_data:])
    dfact_pred ['High'] = df[upper][train_data:]
    dfact_pred ['Low Predict'] = pred_lower
    dfact_pred ['High Predict'] = pred_upper

    dfpred = pd.DataFrame(columns=['Lower Predict', 'Upper Predict', 'MDE^2'], index=range((len(df)),(len(df))+5))  
    MDE2(df[lower][train_data:], pred_lower[:test_data], df[upper][train_data:], pred_upper[:test_data] ,test_data)
    dfpred['Lower Predict'][0:5] = pred_lower[test_data:test_data+5]
    dfpred['Upper Predict'][0:5] = pred_upper[test_data:test_data+5]
    dfpred['MDE^2'][0:5] = "-"
    dfpred['MDE^2'][(len(df))] = round(MDE2(df[lower][train_data:], pred_lower[:test_data], df[upper][train_data:], pred_upper[:test_data] ,test_data),4)
    
    # plot graph
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()

    plt.plot(df[lower], linewidth=3, color='#60bd81')
    plt.plot(df[upper],linewidth=3, color='#3d90f5')

    y = np.arange(train_data,(len(df)))
    for idx, val in dfact_pred.iterrows():
        plt.plot([y[idx-train_data], y[idx-train_data]], [val['Low Predict'], val['High Predict']] , marker='', linewidth=4, alpha=1.0, color='#e31414', solid_capstyle='round')

    for idx, val in dfpred.iterrows():
        plt.plot([idx, idx], [val['Lower Predict'], val['Upper Predict']] , marker='', linewidth=4, alpha=1.0, color='#ff9100', solid_capstyle='round')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)  # change width 

    low_patch = mpatches.Patch(color='#60bd81', label='Lower Actual') 
    high_patch = mpatches.Patch(color='#3d90f5', label='Upper Actual') 
    forecast_patch = mpatches.Patch(color='#e31414', label='Forecast') 
    future_patch = mpatches.Patch(color='#ff9100', label='Future')

    plt.xlabel('Day',fontsize=20)
    plt.ylabel('Price',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(handles=[low_patch,high_patch,forecast_patch,future_patch], loc='best', fontsize=14,framealpha=1.0, frameon=True)
    
    st.session_state['save_pic'] = io.BytesIO()
    plt.savefig(st.session_state['save_pic'], format='png')

    st.session_state['Graph'] = fig
    
    st.session_state['summary_low_model'] = lower_model_fit.summary()
    st.session_state['summary_high_model'] = Upper_Model_fit.summary()

    print(st.session_state['summary_low_model'] )
    text = ("Lower Model\n"+lower_model_fit.summary().as_text()+"\n\n\n\nUpper Model\n"+Upper_Model_fit.summary().as_text())
    
    with io.StringIO() as file:
        file.write(text)
        st.session_state['summary_model']  = file.getvalue()

    st.session_state['dfpred'] = dfpred
    st.session_state['run_complete'] = True
    
    
#---------------------------------------------------------------------------#
#    Set Parameter
def set_params(df, list_col): 
    global start_button, upper, lower, init_temp, stop, n_test, test_data, train_data
    col1, col2, col3 = st.columns(3 ,gap="large")

    with col1:
        st.subheader("Predict option")
        st.markdown("Predict column")
        lower = st.selectbox('Lower Column:', list_col) 
        upper = st.selectbox('Upper Column:', list_col) 
        
            #period = [1,3,12]
            #n_period = st.selectbox('Number of Period', period) # ใส่ help

    with col2:
        st.subheader("Choose SA parameters")
        init_temp =st.number_input('Initial Temperature', step=1, min_value=10, value=1000) # ใส่ help
        stop = st.number_input('The number of repeated solutions', value=500, help=' Number of iterations for stopping criteria when solution does not improve.', min_value=0, step=1)
        
    with col3:
        num_data = len(df)
        num_datatest = st.radio('**Amount of Test Data**',["Percentage","Number of data points"])
        if num_datatest == "Percentage":
            n_test = st.number_input('Enter your number of test data', value=30, step=1, min_value=1, max_value=100)
            test_data = int(num_data*(n_test/100))
            train_data = num_data-test_data
        if num_datatest == "Number of data points":
            n_test = st.number_input('Enter your number of test data', value=30, step=1, min_value=1)
            test_data = n_test
            train_data = num_data-test_data

        start_button = st.button("Start",type="primary",on_click=start_it)


#---------------------------------------------------------------------------#
#input file
uploaded_file = st.file_uploader("# Choose a file")

if uploaded_file is None:
    st.info("Awaiting for file to be uploaded.")

    if 'press_example_button' not in st.session_state:
        st.session_state['press_example_button'] = False

    if st.button('Press to use Example Dataset'):
        st.session_state['press_example_button'] = True

    if st.session_state['press_example_button']:  
        Low =[49.25,50,46.75,44.5,45,45.75,44.75,43.5,43.75,43.75,44.25,42.75,44.25,45.25,46.75,47,47.25,46.75,48,48.75,48.5,49,49.25,
                49,49.25,50,50,50.25,49.25,48.75,49.5,49,48.75,49,49.25,50.25,51.25,50.5,51.25,50.75,52.25,52,50.75,51,51.75,52.5,52.75,54,53.75,53,53.5,52.5
                ,51.25,51.5,50,50,50.25,49,48,46.5,46.25,45.75,43.75,45.25,46.75,46.25,45,43.75,42.75,44.5,45.5,46.25,46.25,46,46.5,46.75,46.75,46.75,46.25,45,45.25,45.75,44.75,44.75,42.75
                ,43.5,45.5,45.75,47.5,47,47.25,46,45,43.75,43.25,40,41.25,42.5,43,42,41.75,41.5,44.25,43.75,42.75,42,41.75,41.5,41.75,42,40.5,40,40.75
                ,39.5,40.25,40.25,39,39.25,39.5,40.25,40.75,40.75,40.75,40.5,40,40,40.25,41.5,41.75,41.75,42,41.75,41,40.75,39.5,38.5,39,38.5,37.75,35.25,36,36.5,36.5]
        
        High =[52.25,51.75,51.25,46.75,46.75,47.5,45.75,45.5,45.25,45.25,45.25,44.75,45.75,47.25,48,48.25,48.25,48,49.75,50.25,49.75,50,50.25,50.25,50.75,51.25,53.25,51.25,51.25
                ,50.25,51,50.75,49.5,50.25,50.25,51.75,52.25,51.5,52,52.5,53.25,53,52.5,53,52.75,53.5,53.5,54.75,54.5,53.75,54.5,53.75,52.5,52,51.75,51.75,51.5,50,49.75
                ,48.2,48.25,47.25,46.75,47,48.25,47.75,47.25,45,45,46.5,46.5,47.5,47.5,47.25,47.5,47.5,48,47.5,47.25,46.75,46.25,46.5,45.75,45.5,45.25,44.75,47.25,47,48.25,48,48.25,47,45.75,45.5,44.5,44.5
                ,42.,43.75,43.75,42.75,42.75,44,45,44.5,44,43.5,42.75,43,42.75,42.5,42,41,41.5,41.5,41,40.75,40.25,39.75,40.75,41.5,41.5,41.75,41.25,41.5,40.75,40.5
                ,42,42.75,42.5,42.75,43.5,42.75,42.25,41.75,41.25,40,39.75,39.5,39.25,38,37.5,38,37.75]
        
        data = list(zip(Low,High))
        df = pd.DataFrame(data,columns=['Low','High'])
        expander_ex = st.expander('This is an example for interval time series data.')
        expander_ex.write(df)
        list_col = df.select_dtypes(include=['float','int']).columns
        set_params(df, list_col)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    see_input_data_expander = st.expander("See your data.")
    see_input_data_expander.write(df)
    list_col = df.select_dtypes(include=['float','int']).columns
    set_params(df, list_col)
    
st.divider()

progess_box = st.empty()
if 'progress_bar_value' not in st.session_state:
    st.session_state['progress_bar_value'] = 0
my_bar = st.progress(text=f"Progress {st.session_state['progress_bar_value']}%", value=st.session_state['progress_bar_value'])

#progress_arae = st.text_area('process')

tab1, tab2, tab3 = st.tabs(["Graph", "Predicted Value", "Summary"])

with tab1:
    #st.write('**Interval Time Series Graph**')
    if 'Graph' not in st.session_state:
        st.session_state['Graph'] = plt.figure(figsize=(20,10))
    st.pyplot(st.session_state['Graph'])

with tab2:
    #result_table = st.table(dfpred)
    if 'dfpred' not in st.session_state:
        st.session_state['dfpred'] = pd.DataFrame()
    st.table(st.session_state['dfpred'])

with tab3:
    if 'summary_low_model' not in st.session_state:
        st.session_state['summary_low_model'] = ""
    if 'summary_high_model' not in st.session_state:
        st.session_state['summary_high_model'] = ""
    
    st.text_area(label="**Low Model**", value=st.session_state['summary_low_model'], disabled=True, height=550)
    st.text_area(label="**High Model**", value=st.session_state['summary_low_model'],disabled=True, height=550)

if 'run_complete' not in st.session_state:
    st.session_state['run_complete'] = False
if 'save_pic' not in st.session_state:
    st.session_state['save_pic'] = io.BytesIO()
if 'summary_model' not in st.session_state:
    st.session_state['summary_model'] = io.BytesIO()

st.download_button(label="Download Image",data=st.session_state['save_pic'],file_name="Graph_from_IStockcast.png", mime="image/png",disabled=not st.session_state['run_complete'])
st.download_button(label="Download Forecast Data ", data=st.session_state['dfpred'].to_csv(), file_name='Forecast_data_from_IStockcast.csv',mime='text/csv',disabled=not st.session_state['run_complete'])
st.download_button(label="Download Summary Model", data=st.session_state['summary_model'], file_name='Summary_model_from_IStockcast.txt',mime='text/txt',disabled=not st.session_state['run_complete'])


