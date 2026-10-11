# NOTE: this cell is NOT meant to be run in Colab. It is the content of a file
#       named app.py, which Streamlit Community Cloud runs for you. Copy the whole
#       cell into app.py; the comments can stay, they do no harm.
#       Tip: put  %%writefile app.py  as the very first line of this cell and run it;
#       Colab then saves the cell as a file instead of executing it.
#
# Streamlit turns a Python script into a web page: every st.something(...) call
# puts one element on the page, in the order the calls are made.
#   streamlit as st : the web-page toolkit.
#   pipeline        : imported but never used below (see the tip at the end).
#   the rest        : same as in the test above.
import streamlit as st
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np


# Streamlit re-runs this whole file from top to bottom every time the visitor does
# something on the page, such as typing a sentence and pressing Enter.
def main():


    # st.title : large heading.   st.write : ordinary text.
    # >>> MODIFY: the wording, to describe your own app.
    st.title("yelp2024fall Test")
    st.write("Enter a sentence for analysis:")

    # st.text_input draws a text box and returns what the visitor typed. It returns
    # an empty string until they type something and press Enter, so the code under
    # `if user_input:` runs only once there is text to classify.
    # The "" is the box's label, left empty here. Streamlit prefers a real label,
    # e.g. st.text_input("Your sentence").
    user_input = st.text_input("")
    if user_input:
        # Approach: AutoModel
        # Download the fine-tuned model from the Hub (first time only; afterwards
        # it is read from the app's disk) and load it.
        # >>> MODIFY: "<<Model name>>" -> your own, e.g. "your_username/CustomModel_yelp".
        #     The model must be PUBLIC on the Hub; this app supplies no login token.
        # NOTE: because loading sits inside `if user_input:`, the model is loaded
        #       again for every sentence submitted, which makes each answer slow.
        #       The tip at the end of this cell shows how to load it only once.
        model2 = AutoModelForSequenceClassification.from_pretrained("<<Model name>>",
                                                                    num_labels=5)
        # As above: this can also come from your own repository.
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # The same prediction steps as in the first test, using the visitor's text
        # instead of a fixed review.
        inputs = tokenizer(user_input,
                        padding=True,
                        truncation=True,
                        return_tensors='pt')

        outputs = model2(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()
        # Get the index of the largest output value
        max_index = np.argmax(predictions)
        # Show the result on the page. The label is 0 to 4, meaning 1 to 5 stars.
        # >>> MODIFY: e.g. st.write(f"Predicted rating: {max_index + 1} stars")
        st.write(f"result (AutoModel) - Label: {max_index}")


# Standard Python: call main() when this file is run as a program, which is how
# Streamlit runs it.
if __name__ == "__main__":
    main()

# Tip: load the model once instead of on every submission. Put this above main()
# and, inside main(), replace the two from_pretrained lines with
#     model2, tokenizer = load_model()
#
# @st.cache_resource      # Streamlit keeps what the function returns between re-runs
# def load_model():
#     model = AutoModelForSequenceClassification.from_pretrained("your_username/your_model")
#     tok = AutoTokenizer.from_pretrained("your_username/your_model")
#     return model, tok
#
# Tip: `pipeline` (imported above, unused) does tokenizing, the model call and
# softmax in one step:
#     classifier = pipeline("text-classification", model="your_username/your_model")
#     classifier("some text")   ->   [{'label': 'LABEL_3', 'score': 0.41}]
