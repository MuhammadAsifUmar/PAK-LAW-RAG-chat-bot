# create_dataset.py
from langsmith import Client
import pandas as pd
import os

# ✅ Initialize LangSmith client
client = Client()

# ✅ Read Excel file
excel_file = "F:\project1\project\chatbot_evaluation1_template.xlsx"


if not os.path.exists(excel_file):
    raise FileNotFoundError(f"Excel file '{excel_file}' not found!")

print(f"📖 Reading {excel_file}...")
df = pd.read_excel(excel_file)

# ✅ Validate columns
required_columns = ["question", "expected_answer"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Excel must have columns: {required_columns}")

print(f"✅ Found {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# ✅ Create dataset
dataset_name = "ppc-crpc-qa-eval"

# Check if dataset exists
try:
    existing_datasets = list(client.list_datasets(dataset_name=dataset_name))
    if existing_datasets:
        print(f"⚠️  Dataset '{dataset_name}' already exists. Deleting...")
        for ds in existing_datasets:
            client.delete_dataset(dataset_id=ds.id)
except:
    pass

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Evaluation dataset for PPC/CrPC chatbot (from Excel)"
)

print(f"✅ Created dataset '{dataset_name}'")

# ✅ Add examples from Excel
for idx, row in df.iterrows():
    question = str(row["question"]).strip()
    expected_answer = str(row["expected_answer"]).strip()
    
    # Optional columns
    expected_sections = []
    expected_book = ""
    
    if "expected_sections" in df.columns and pd.notna(row["expected_sections"]):
        # Handle comma-separated sections: "302,303" or "302, 303"
        sections_str = str(row["expected_sections"]).strip()
        expected_sections = [s.strip() for s in sections_str.split(",")]
    
    if "expected_book" in df.columns and pd.notna(row["expected_book"]):
        expected_book = str(row["expected_book"]).strip()
    
    # Create example
    client.create_example(
        inputs={"question": question},
        outputs={
            "expected_answer": expected_answer,
            "expected_sections": expected_sections,
            "expected_book": expected_book
        },
        dataset_id=dataset.id
    )
    
    print(f"  ✅ Added example {idx + 1}: {question[:50]}...")

print(f"\n🎉 Dataset created successfully!")
print(f"📊 Total examples: {len(df)}")
print(f"🔗 View at: https://smith.langchain.com/")