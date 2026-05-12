from rag_chain import ask_question
import pandas as pd
import time
import os

# 1. File ka naam sahi se set karein
file_name = "chatbot_evaluation_template.xlsx"

# 2. Check karein ke file folder mein hai ya nahi
if not os.path.exists(file_name):
    print(f"❌ Error: '{file_name}' folder mein nahi mili!")
    print("Folder mein ye files mojud hain:")
    print(os.listdir(".")) # Folder ki saari files print karega taake spelling check ho sakay
else:
    try:
        print(f"📂 Reading file: {file_name}...")
        df = pd.read_excel(file_name)
        
        # Check agar 'Question' column mojud hai
        if "Question" not in df.columns:
            print("❌ Error: Excel file mein 'Question' naam ka column nahi mila!")
        else:
            answers = []
            for q in df["Question"]:
                # Khali rows ko skip karne ke liye
                if pd.isna(q):
                    continue
                    
                print(f"🚀 Running Query: {q}")
                try:
                    ans = ask_question(q)
                    answers.append(ans)
                    print("✅ Answer Received.")
                except Exception as e:
                    print(f"⚠️ Query fail hui: {e}")
                    answers.append("Error in generation")
                
                time.sleep(2) # Groq API ki safety ke liye

            # Bot answers save karo
            df["Bot Answer"] = pd.Series(answers)
            output_file = "results_filled.xlsx"
            df.to_excel(output_file, index=False)
            
            print(f"\n✨ Done! Mubarak ho Faisal bhai. Result yahan save hai: {output_file} ✅")

    except Exception as e:
        print(f"❌ Excel parhne mein masla aya: {e}")