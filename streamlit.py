import streamlit as st
import pandas as pd
from transpose import Transpose
import requests
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

#------------------------------------------------------#
#Write the necessary functions
def get_rook_reward():
    url = "https://api.rook.fi/api/v1/coordinator/userClaims?user=0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE"
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
token_metadata = pd.json_normalize(api.token.tokens_by_contract_address([contract_address_usdc, contract_address_dai, contract_address_frax, contract_address_usdt, contract_address_weth, contract_address_rook]).to_dict())[['contract_address', 'name', 'symbol', 'decimals']] 
wallet_balances = pd.json_normalize(api.token.tokens_by_owner(testing_wallet).to_dict())
def token_amount(df):
    try:
        return df['balance']/pow(10, df['decimals'])
    except:
        return df['quantity']/pow(10, df['decimals'])
    
def create_trade_path(df):
    return f"{df['symbol_x']}-->{df['symbol_y']}"
token_transfers_all = pd.json_normalize(api.token.transfers_by_account(testing_wallet, limit=500).to_dict()).merge(token_metadata, how='left', left_on='contract_address', right_on='contract_address').assign(token_amount = lambda x: token_amount(x))
names_dict = {'0x6d956A6Aaca9BB7A0e4D34b6924729F856c641dE':'Testing Wallet'}
token_transfers = token_transfers_all[['transaction_hash', 'timestamp', 'contract_address','symbol', 'token_amount', 'from', 'to']].rename(columns={'transaction_hash':'tx_hash', 'contract_address':'token_contract_address', 'from':'sender', 'to':'receiver'}).replace({'sender': names_dict, 'receiver':names_dict}).sort_values('timestamp', ascending=False)

token_transfer_from = token_transfers[token_transfers['sender']=='Testing Wallet']
token_transfer_to = token_transfers[token_transfers['receiver']=='Testing Wallet']
token_sent = token_transfer_from.merge(token_transfer_to, how='left', left_on=['tx_hash'], right_on=['tx_hash'])[['tx_hash', 'timestamp_x', 'token_contract_address_x', 'symbol_x', 'token_amount_x', 'sender_x', 'receiver_x', 'token_contract_address_y', 'symbol_y', 'token_amount_y']].dropna(subset=['token_contract_address_y'])
token_sent['path'] = token_sent.apply(create_trade_path, axis=1)
token_sent['token_amount_x']=token_sent['token_amount_x'].astype('float64')

scam_coins = wallet_balances[wallet_balances['contract_address'] == '0x1f068a896560632a4d2E05044BD7F03834f1A465'].index
bal = wallet_balances.drop(scam_coins).assign(amount = lambda x: token_amount(x))
#--------------------------------------------------------------------------------------------#




st.set_page_config(layout="wide")
st.title("Rook Multisig Testing Wallet")
st.write("Learn More: [KIP-30](https://forum.rook.fi/t/kip-30-temporarily-empower-and-fund-a-strategy-testing-multisig/395)")



rook_earned_col, rook_claimed_col, rook_claimable_col,  = st.columns(spec=3, gap="medium")

st.header("Current Balances")

stablecol_1, stablecol_2, stablecol_3, stablecol_4 = st.columns(spec=4, gap="small")

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
fig.update_traces(hole=.4, hoverinfo="label+percent+value")
fig.update_layout(
    title_text="Current Wallet Testing Wallet Balances & Volume",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Balance', x=0.2, y=0.50, font_size=25, showarrow=False),
                 dict(text='Volume', x=0.80, y=0.5, font_size=25, showarrow=False)],
    autosize = True,
    width=900,
    height=650,)
st.plotly_chart(fig,use_container_width=True)


stable_vol_1, stable_vol_2, stable_vol_3, stable_vol_4 = st.columns(spec=4, gap="small")
rook_reward=get_rook_reward()




# with st.expander("Current Balances"):





trading_details_expander = st.expander("Trading Details")
# st.line_chart()
# st.table(token_sent[(token_sent['symbol_x']=='USDC') | (token_sent['symbol_y']=='USDC')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x',ascending=False))
token_sent['cumsum']=token_sent.groupby(['timestamp_x', 'path'])['token_amount_x'].cumsum()
st.dataframe(token_sent[['timestamp_x', 'path', 'token_amount_x', 'cumsum']])
with rook_earned_col:
    st.metric(label="Total $ROOK Earned", value=f"{rook_reward[0]:.2f} $ROOK")
with rook_claimed_col:
    st.metric(label="Total $ROOK Claimed", value=f"{rook_reward[1]:.2f} $ROOK")
with rook_claimable_col:
    st.metric(label="Claimable $ROOK", value=f"{rook_reward[2]:.2f} $ROOK")

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
    st.dataframe(token_sent[(token_sent['symbol_x']=='USDT') | (token_sent['symbol_y']=='USDT')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x', ascending=False))

with stablecol_4:
    st.subheader("FRAX")
    frax_balance = float(bal[bal['name'] == 'Frax']['amount'])
    change = (frax_balance-150000)/150000
    st.metric(label="FRAX", value = f"${frax_balance:,.0f}", delta =f"{change:.2%}")

with stable_vol_4:
    total_frax_volume = token_sent[token_sent['symbol_x']=='FRAX']['token_amount_x'].sum()
    st.metric(label="FRAX Volume", value=f"${total_frax_volume:,.0f}")
    st.dataframe(token_sent[(token_sent['symbol_x']=='FRAX') | (token_sent['symbol_y']=='FRAX')][['timestamp_x', 'path', 'token_amount_x']].sort_values(by='timestamp_x', ascending=False))



with trading_details_expander:
    cum_sum = token_sent.groupby(['timestamp_x', 'path'])['token_amount_x'].cumsum()
    st.dataframe(cum_sum)
    stable_details_col1, stable_details_col2, stable_details_col3, stable_details_col4 = st.columns(spec=4, gap="medium")

with stable_details_col1:
    st.write("USDC")
    st.metric(label="# of Trade", value=50)

with stable_details_col2:
    st.write("DAI")
    st.metric(label="# of Trade", value=50)


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