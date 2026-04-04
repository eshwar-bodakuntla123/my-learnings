import pandas as pd

# Load CSV
data = pd.read_csv("RuleBasedImplementation/faq.csv")

def chatbot(user_input):

    for i, row in data.iterrows():
        if row['question'].lower() in user_input.lower():
            return row['answer']
    return "Sorry, I don't know the answer yet."



if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("Bot:", chatbot(query))