import joblib


model = joblib.load("cyberbullying_model.joblib")
vectorizer = joblib.load("cyber_vectorizer.joblib")

print("\n===============================")
print("  Cyberbullying Detection Test ")
print("===============================")

while True:
    text = input("\nEnter text to analyze (or type 'exit' to quit): ")

    if text.lower() == "exit":
        print("\nExiting... Goodbye! 👋")
        break

    # Transform input
    transformed = vectorizer.transform([text])
    result = model.predict(transformed)[0]

    # Output Result
    print("\nPrediction Result:")
    if result == 1:
        print("🚨 ALERT: Potential Cyberbullying Detected - Action Recommended.")
    else:
        print("🟩 Status: No signs of cyberbullying detected.")
