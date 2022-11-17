import streamlit as st
import pandas as pd
from transpose import Transpose
import requests
from datetime import datetime, timezone, date
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dotenv import load_dotenv
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import time

from pandas.io.json import json_normalize
import json

from datetime import date

today = date.today()

st.set_page_config(page_title = "Rook Stablecoin Testing Wallet", layout="wide")



#------------------------------------------------------#
#Write the necessary functions
@st.cache
def get_rook_reward():
    url = "https://api.rook.finance/api/v1/coordinator/userClaims?user=0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE"
    response = (requests.get(url).json())
    earning_to_date = (response['latestCommitment']['earningsToDate'])/pow(10,18)
    total_claimed = (response['totalClaimed'])/pow(10,18)
    claimable = earning_to_date - total_claimed
    return earning_to_date, total_claimed, claimable



#------------------------------------------------------------------------#
#Constants

testing_wallet = str.lower("0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE")
multisig_1 = str.lower("0x722f531740937fc21A2FaC7648670C7f2872A1c7")
multisig_2 = str.lower("0xDAAf6D0ab259053Bb601eeE69880F30C7d8e5853")
multisig_3 = str.lower("0x3C3ca4E5AbF0C1Bec701375ff31342d90D8C435E")

contract_address_usdc = str.lower("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
contract_address_dai = str.lower('0x6B175474E89094C44Da98b954EedeAC495271d0F')
contract_address_frax = str.lower('0x853d955aCEf822Db058eb8505911ED77F175b99e')
contract_address_usdt = str.lower('0xdAC17F958D2ee523a2206206994597C13D831ec7')
contract_address_rook = str.lower('0xfA5047c9c78B8877af97BDcb85Db743fD7313d4a')
contract_address_weth = str.lower('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2')
#-----------------------------------------------------------------------------------#

#-----Get Current Wallet Balances-----#
load_dotenv()
TRANSPOSE_API_KEY = os.environ.get('TRANSPOSE_API_KEY')
api=Transpose(TRANSPOSE_API_KEY)


#use transpose API to pull the data
token_metadata=pd.DataFrame()
contract_list = [contract_address_usdc, contract_address_dai, contract_address_frax, contract_address_usdt, contract_address_weth, contract_address_rook]
for contract in contract_list:
    temp_list = []
    temp_metadata = api.bulk_request(api.token.tokens_by_contract_address(contract), requests_per_second=0.33)
    for tmp in temp_metadata:
        temp_list.append(tmp.to_dict())
    temp_metadata = pd.json_normalize(temp_list)[['contract_address', 'name', 'symbol', 'decimals']]
    token_metadata =  pd.concat([token_metadata, temp_metadata], axis=0)
    time.sleep(1)
wallet_balances = pd.json_normalize(api.token.tokens_by_owner(testing_wallet).to_dict())
time.sleep(1)

def token_amount(df):
    try:
        return df['balance']/pow(10, df['decimals'])
    except:
        return df['quantity']/pow(10, df['decimals'])

def create_trade_path(df):
    return f"{df['symbol_x']}-->{df['symbol_y']}"


# token_transfers_all = api.bulk_request(api.token.transfers_by_account(testing_wallet, limit=500))  ##Get the token transfers using bulk request
token_transfers_all = api.bulk_request(api.token.transfers_by_account(testing_wallet, limit=500), requests_per_second=0.33)
transfers_list =[]
for tt in token_transfers_all:
    transfers_list.append(tt.to_dict())
df_token_transfers_all= pd.json_normalize(transfers_list)     #transform into df from dictionary
# print('###--------------------------------------------------------###')
# print('###--------------------------------------------------------###')

# print(token_transfers_all)

# print('###--------------------------------------------------------###')
# print('###--------------------------------------------------------###')






                                        
df_token_transfers_all_w_metadata = df_token_transfers_all.merge(token_metadata, how='left', left_on='contract_address', right_on='contract_address').assign(token_amount = lambda x: token_amount(x))

names_dict = {'0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE':'Testing Wallet'}
token_transfers = df_token_transfers_all_w_metadata[['transaction_hash', 'timestamp', 'contract_address','symbol', 'token_amount', 'from', 'to']].rename(columns={'transaction_hash':'tx_hash', 'contract_address':'token_contract_address', 'from':'sender', 'to':'receiver'}).replace({'sender': names_dict, 'receiver':names_dict}).sort_values('timestamp', ascending=False)

token_transfer_from = token_transfers[token_transfers['sender']=='Testing Wallet']
token_transfer_to = token_transfers[token_transfers['receiver']=='Testing Wallet']
token_sent = token_transfer_from.merge(token_transfer_to, how='left', left_on=['tx_hash'], right_on=['tx_hash'])[['tx_hash', 'timestamp_x', 'token_contract_address_x', 'symbol_x', 'token_amount_x', 'sender_x', 'receiver_x', 'token_contract_address_y', 'symbol_y', 'token_amount_y']].dropna(subset=['token_contract_address_y'])
token_sent['path'] = token_sent.apply(create_trade_path, axis=1)
token_sent['token_amount_x']=token_sent['token_amount_x'].astype('float64')
token_sent['date']=pd.to_datetime(token_sent['timestamp_x']).dt.date
token_sent['key']=1

token_sent['date']=token_sent['date'].astype('datetime64')
scam_coins = wallet_balances[wallet_balances['contract_address'] == '0x1f068a896560632a4d2E05044BD7F03834f1A465'].index
bal = wallet_balances.drop(scam_coins).assign(amount = lambda x: token_amount(x))
#--------------------------------------------------------------------------------------------#

#-------------------------------------
#-------------------------------------Create df with daily values
date_range = pd.date_range(pd.to_datetime(token_sent['timestamp_x']).dt.date.min(), date.today(), inclusive="both").to_frame(index=False)
date_range.columns = ['date']
date_range['key']=1

tokens=pd.DataFrame(token_sent['symbol_x'].unique())
token_paths = pd.DataFrame(token_sent[['symbol_x', 'path']])
tokens['key']=1
token_paths['key']=1
token_paths=token_paths.groupby(['symbol_x', 'path'])['key'].mean().reset_index() 

daily_tokens = date_range.merge(tokens, how='left', on='key')
daily_tokens = daily_tokens.merge(token_paths, how= 'left', left_on=['key', 0], right_on=['key', 'symbol_x']).drop(['key', 0], axis=1)

token_sent_sum = token_sent.groupby(['date', 'symbol_x', 'path'])['token_amount_x'].sum().reset_index().sort_values('date')
token_sent_sum=daily_tokens.merge(token_sent_sum, how='left', left_on=['date', 'symbol_x', 'path'], right_on=['date', 'symbol_x', 'path']).fillna(0)
token_sent_sum['cumsum'] = token_sent_sum.groupby(['symbol_x', 'path'])['token_amount_x'].cumsum()
token_sent_sum=token_sent_sum[token_sent_sum['cumsum']>0].sort_values(['date', 'symbol_x', 'path'])

# st.dataframe(token_sent_sum)


st.title("Rook Multisig Testing Wallet")
st.write("Learn More: [KIP-30](https://forum.rook.fi/t/kip-30-temporarily-empower-and-fund-a-strategy-testing-multisig/395)")

#-------------------------------------------------------------------------#
# Show the amount of Rook that has been earned using columns
rook_earned_col, rook_claimed_col, rook_claimable_col,cumulative_volume  = st.columns(spec=4, gap="small")

st.header("ROI")

roi_1,roi_2, roi_3 = st.columns(spec=3,gap="small")

st.header("Current Balances")

stablecol_1, stablecol_2, stablecol_3, stablecol_4 = st.columns(spec=4, gap="small")

rebate_comb = pd.DataFrame(columns=['txHash','userRookRebate','rookPrice'])
for i in range(1,1000000):
    try:
        rebate_url = "https://api.rook.finance/api/v1/trade/fills?makerAddresses=0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE&page="+str(i)+"&size=100"
        response = requests.get(rebate_url)
        parsed = json.loads(response.content)
        rebate_df = json_normalize(parsed['items'])
        rebate_df = rebate_df[['txHash','userRookRebate','rookPrice']]
        rebate_comb = rebate_comb.append(rebate_df)
    except:
        break

rebate_comb['rewards'] = rebate_comb['userRookRebate'] * rebate_comb['rookPrice']
time_between = date.today() - date(1981, 12, 2)
time_between.days
with roi_1:
    st.metric(label="Total USD Earned", value=f"{rebate_comb['rewards'].sum():.2f} $USD")
    
with roi_2:
    st.metric(label="ROI", value=f"{rebate_comb['rewards'].sum()/600000:.2f} %ROI")
    
with roi_3:
    st.metric(label="Effective API", value=f"{rebate_comb['rewards'].sum()/600000 * (365/time_between.days):.2f} % Effective ROI")
        
        
#-----------------------------------------------------------------------------------#
#------------------Create Pie Charts for Showing Wallet Balances--------------------#
labels = ["USDC", "USDT", "DAI", "FRAX"]
wallet_balances = [f"{float(bal[bal['name'] == 'USD Coin']['amount']):.0f}",f"{float(bal[bal['name'] == 'Tether USD']['amount']):.0f}", f"{float(bal[bal['name'] == 'Dai Stablecoin']['amount']):.0f}", f"{float(bal[bal['name'] == 'Frax']['amount']):.0f}"]
stablecoin_volume = [token_sent[token_sent['symbol_x']=='USDC']['token_amount_x'].sum(), token_sent[token_sent['symbol_x']=='USDT']['token_amount_x'].sum(), token_sent[token_sent['symbol_x']=='DAI']['token_amount_x'].sum(), token_sent[token_sent['symbol_x']=='FRAX']['token_amount_x'].sum()]
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=wallet_balances, name="Stablecoin Balances"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=stablecoin_volume, name="Trading Volume"),
              1, 2)
# Use `hole` to create a donut-like pie chart
fig.update_traces(text=labels, hole=.4, hoverinfo="label+percent+value")
fig.update_layout(
    title_text="Current Wallet Testing Wallet Balances & Volume",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Balance', x=0.2, y=0.50, font_size=25, showarrow=False),
                 dict(text='Volume', x=0.80, y=0.5, font_size=25, showarrow=False)],
    autosize = True,
    width=900,
    height=650,)
st.plotly_chart(fig,use_container_width=True)
#----------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------
#----------------------------Create stacked area chart of volume-----------


fig= px.area(token_sent_sum, 'date', 'cumsum', color='path', line_group = 'symbol_x')
st.plotly_chart(fig, use_container_width=True)



#-----------------------------------------------------------------------------
stable_vol_1, stable_vol_2, stable_vol_3, stable_vol_4 = st.columns(spec=4, gap="small")
rook_reward=get_rook_reward()

trading_details_expander = st.expander("Trading Details", expanded=True)
# st.line_chart()
# st.table(token_sent[(token_sent['symbol_x']=='USDC') | (token_sent['symbol_y']=='USDC')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x',ascending=False))
token_sent['cumsum']=token_sent.groupby(['timestamp_x', 'path'])['token_amount_x'].cumsum()



# st.dataframe(token_sent[['timestamp_x', 'path', 'token_amount_x', 'cumsum']])
with rook_earned_col:
    st.metric(label="Total $ROOK Earned", value=f"{rook_reward[0]:.2f} $ROOK")
with rook_claimed_col:
    st.metric(label="Total $ROOK Claimed", value=f"{rook_reward[1]:.2f} $ROOK")
with rook_claimable_col:
    st.metric(label="Claimable $ROOK", value=f"{rook_reward[2]:.2f} $ROOK")

with cumulative_volume:
    total_volume = token_sent['token_amount_x'].sum()
    st.metric(label="Total Volume", value=f"${total_volume:,.0f}")    

with stablecol_1:
    st.subheader("USDC")
   
    token_sent[['path','tx_hash']].groupby('path').count()
    
    
    usdc_balance = float(bal[bal['name'] == 'USD Coin']['amount'])

    change = (usdc_balance-150000)/150000
    
    st.metric(label="USDC Balance", value = f"${usdc_balance:,.2f}", delta =f"{change:.2%}")


    # st.table((token_sent[token_sent['symbol_x']=='USDC') | ][['timestamp_x','path', 'token_amount_x']].sort_values('timestamp_x', ascending=False))
with stable_vol_1:
    total_usdc_volume = token_sent[token_sent['symbol_x']=='USDC']['token_amount_x'].sum()
    st.metric(label="USDC Volume", value=f"${total_usdc_volume:,.0f}")
    st.dataframe(token_sent[(token_sent['symbol_x']=='USDC') | (token_sent['symbol_y']=='USDC')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x', ascending=False), )

with stablecol_2:
    st.subheader("DAI")
    dai_balance = float(bal[bal['name'] == 'Dai Stablecoin']['amount'])
    change = (dai_balance-150000)/150000
    st.metric(label="DAI", value = f"${dai_balance:,.2f}", delta =f"{change:.2%}")

    # st.table(token_sent[token_sent['symbol_x']=='DAI'][['timestamp_x','path', 'token_amount_x']].sort_values('timestamp_x', ascending=False))
    # with st.expander("Recent Trades"):
    # st.write("# Trades")
with stable_vol_2:
    total_dai_volume = token_sent[token_sent['symbol_x']=='DAI']['token_amount_x'].sum()
    st.metric(label="DAI Volume", value=f"${total_dai_volume:,.0f}")
    st.dataframe(token_sent[(token_sent['symbol_x']=='DAI') | (token_sent['symbol_y']=='DAI')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x', ascending=False))

with stablecol_3:
    st.subheader("USDT")

    usdt_balance = float(bal[bal['name'] == 'Tether USD']['amount'])
    change = (usdt_balance-150000)/150000
    st.metric(label="USDT", value = f"${usdt_balance:,.2f}", delta =f"{change:.2%}")

with stable_vol_3:
    total_usdt_volume = token_sent[token_sent['symbol_x']=='USDT']['token_amount_x'].sum()
    st.metric(label="USDT Volume", value=f"${total_usdt_volume:,.0f}")
    usdt_df = token_sent[(token_sent['symbol_x']=='USDT') | (token_sent['symbol_y']=='USDT')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x', ascending=False)
    
    st.dataframe(usdt_df.style.format(precision=0, formatter={'token_amount_x':"${:,.0f}"}))

with stablecol_4:
    st.subheader("FRAX")
    frax_balance = float(bal[bal['name'] == 'Frax']['amount'])
    change = (frax_balance-150000)/150000
    st.metric(label="FRAX", value = f"${frax_balance:,.0f}", delta =f"{change:.2%}")

with stable_vol_4:
    total_frax_volume = token_sent[token_sent['symbol_x']=='FRAX']['token_amount_x'].sum()
    st.metric(label="FRAX Volume", value=f"${total_frax_volume:,.0f}")
    frax_df = token_sent[(token_sent['symbol_x']=='FRAX') | (token_sent['symbol_y']=='FRAX')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x', ascending=False)
    # pd.set_option('display.float_format', '${:,}'.format)
    # style = frax_df.style.apply(lambda x: "${x:,.0f}".format)
    st.dataframe(frax_df.style.format(precision=0, formatter={'token_amount_x':"${:,.0f}"}))


    
trading_df=token_sent.copy()
trading_df = trading_df[['tx_hash', 'timestamp_x','symbol_x', 'symbol_y', 'path', 'token_amount_x']].sort_values('timestamp_x', ascending=True)
trading_df['cumsum'] = trading_df.groupby(['symbol_x'])['token_amount_x'].cumsum()


trading_df.rename(columns={'tx_hash':'Transaction Hash', 'timestamp_x':'Timestamp', 'path':'Trade Path', 'token_amount_x':'USD Value', 'cumsum':'Token Total'}, inplace=True)

# trading_df.columns=['Transaction Hash', 'Timestamp', 'Trade Path', 'USD Value','Path Total USD Volume']
trading_df.sort_values('Timestamp', ascending=False, inplace=True)
# trading_df['Timestamp'] =pd.to_datetime(trading_df['Timestamp'])
# trading_df.loc[:, 'Timestamp'] = pd.to_datetime(trading_df['Timestamp']).dt.strftime('%Y-%m-%d %X')
# trading_df.loc[:, 'USD Value'] = trading_df['USD Value'].map('${:,.0f}'.format)
# trading_df.loc[:, 'Path Total USD Volume'] = trading_df['Path Total USD Volume'].map('${:,.0f}'.format)

trading_df['Timestamp'] = pd.to_datetime(trading_df['Timestamp'])
fig1 = px.scatter(trading_df, x="Timestamp", y="USD Value", color = "symbol_x", size = 'Token Total')
fig2 = px.bar(trading_df, x="Timestamp", y='USD Value', color='Trade Path')
fig3 = go.Figure(data=fig1.data + fig2.data)
trading_df = trading_df.merge(rebate_comb,how='left',left_on='Transaction Hash',right_on='txHash')
trading_df['Rebate $ Earned'] = trading_df['rookPrice']*trading_df['userRookRebate']
trading_df = trading_df[['Transaction Hash','Timestamp', 'Trade Path', 'USD Value', 'Token Total','Rebate $ Earned','userRookRebate']]


trading_df.rename(columns={'userRookRebate':'Rook Rebate Earned'}, inplace=True)
# st.dataframe(trading_df)
# st.plotly_chart(fig3, use_container_width=True)




# st.plotly_cahrt(token_sent_sum, )
with trading_details_expander:
    
    filter_radio = st.radio('Filter By Stablecoin or Trade Pair', options=['None', 'Stablecoin', 'Trade Path'])
    cum_sum = token_sent.groupby(['timestamp_x', 'path'])['token_amount_x'].cumsum()
    style_format_dict = {'USD Value': "${:,.0f}", 'Path Total USD Volume':"${:,.0f}"}
    
    filtered_trading_df = trading_df.copy()
    filtered_trading_df.rename(columns={'symbol_x':'Token Sold', 'symbol_y': 'Token Purchased'}, inplace=True)

    current_time = datetime.now(timezone.utc)
    last_trade = trading_df.iloc[0]['Timestamp']
    col1, col2, col3 = st.columns((2,2,6), gap='medium')
    difference = str(current_time - last_trade).split('.')
    
    # st.metric(label='Time Since Last Trade', value = f"{difference[0]} {difference[1]}:{difference[3]}")

    gb = GridOptionsBuilder.from_dataframe(trading_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_column('USD Value', type = ["numericColumn", "numberColumnFilter", "customNumericFormat"], 
                        precision=0,
                        valueFormatter="data.'USD Value'.toLocaleString('en-US');")
                        # ={'USD Value':"${:,.0f}"})
    gb.configure_auto_height(autoHeight=False)
    # gb.configure_column('USD Value', other_column_properties={'value': })
    # gb.configure_column('USD Value', type=["currency"], custom_format_string ="${:,.0f}")
    gridOptions = gb.build()        
    
    
    # trading_df = trading_df.rename(columns = {'tx_hash':'Transaction Hash', 'timestamp_x':'Timestamp', 'path':'Trade Path', 'token_amount_x':'USD Value', 'cumsum':'Path Total USD Volume'})
    if filter_radio == 'Stablecoin':
        filter= st.radio('Select Token:', ['USDC', 'USDT', 'DAI', 'FRAX'])
        filter_mask = trading_df['symbol_x']==filter
        
        filtered_trading_df = trading_df[filter_mask]
    elif filter_radio == 'Trade Path':
        filter= st.selectbox('Select Trade Pair from menu', set(trading_df['Trade Path'].tolist()))
        filtered_trading_df = trading_df[trading_df['Trade Path']==filter]
    else:
        filtered_trading_df = trading_df
        
    with col1: st.metric(label='Last Trade', value = filtered_trading_df.iloc[0]['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
    with col2: st.metric(label='Time Since Last Trade', value = difference[0])
    
    # filtered_trading_df.rename(columns={'symbol_x':'Token Sold', 'symbol_y': 'Token Purchased'}, inplace=True)
    # ftd = filtered_trading_df.style.format(precision=0, formatter={'USD Value':"${:,.0f}",
    #                                                                'Token Total':"${:,.0f}"})
    # dtf = trading_df.style.format(precision=0, formatter={'USD Value':"${:,.0f}",
                                                                #    'Token Total':"${:,.0f}"})
    # frax_df.style.format(precision=0, formatter={'token_amount_x':"${:,.0f}"}))
    AgGrid(filtered_trading_df, gridOptions=gridOptions, height=500, width='100%', fit_columns_on_grid_load=True)
    
    @st.cache
    def convert_df(df):
        return df.to_csv()
    csv = convert_df(trading_df)
    
    st.download_button(label='Download All Trades',
                       data=csv,
                       )





with st.sidebar:
    st.header("Multisig Testing Wallet")
    
    st.write("")
    sidebar_expander_contracts = st.expander("Contract Addresses")

with sidebar_expander_contracts:
    st.subheader("Contracts associated with the testing wallet")
    st.write("Multisig Wallet: 0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE")
    st.subheader("Multisig Signers as specified in KIP 30")
    st.write("Signer 1: 0x722f531740937fc21A2FaC7648670C7f2872A1c7")
    st.write("Signer 2: 0xDAAf6D0ab259053Bb601eeE69880F30C7d8e5853")
    st.write("Signer 3: 0x3C3ca4E5AbF0C1Bec701375ff31342d90D8C435E")
