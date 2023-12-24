import os, sys
import pandas as pd
import streamlit as st

path_this = os.path.abspath(os.path.dirname(__file__))
path_engine = os.path.join(path_this, "..")
sys.path.append(path_this)
sys.path.append(path_engine)

from src.inference import Inference

# Create an instance of Inference
inference = Inference()
df = pd.read_csv(os.path.join(path_engine, "data/customer_details.csv"))
cdf = df
cdf = cdf.drop(columns = ['Unnamed: 0'])
customer_id = df["customer_id"].tolist()
customer_list = []
map_customer = {}
for i in df.index:
    customer_list.append(df["name"][i])
    map_customer[df["name"][i]] = df["customer_id"][i]

df = pd.read_csv(os.path.join(path_engine, "data/purchase_history.csv"))
data = df.sort_values(by='purchase_id')
product_list = data.groupby('customer_id').apply(lambda group: group.sort_values(by='purchase_id')['product_id'].tolist()).tolist()

# Create a dictionary to map customer IDs to their product lists
customer_product_dict = dict(zip(customer_id, product_list))

# Function to get the product list for a selected customer
def get_product_list(selected_customer):
    if selected_customer in customer_product_dict:
        # Return the first 5 products for the selected customer
        return customer_product_dict[selected_customer][:5]
    else:
        return []

def main():
    st.title("Next Product Prediction App")
    st.markdown("-----")

    left_pane, right_pane = st.columns(2)
    with left_pane:
        st.write(cdf)
    with right_pane:
        # Load the tokenizer and model
        inference.load_tokenizer()
        inference.load_model()

        selected_user = st.selectbox("Select user or type to search:", customer_list)
        result = get_product_list(map_customer[selected_user])
        selected_products = [inference.id_product_to_name[item] for item in result]

        if st.button("Predict"):
            try:
                # Example of predicting the next item for the selected products
                predicted_product = inference.predict_next_item(selected_products)

                # Display the predicted product
                st.success(f"Predicted next item: {predicted_product}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
