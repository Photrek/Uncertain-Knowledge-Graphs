import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import networkx as nx
import torch
import pyro
import pyro.distributions as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Read the data from CSV
data = pd.read_csv(r'path and name file.csv')

# Creating a directed graph for uncertain knowledge
UKG = nx.DiGraph()

# Adding nodes and edges to the graph
for _, row in data.iterrows():
    UKG.add_node(row['Disease'].lower(), node_type='Disease')
    UKG.add_node(row['Medication'].lower(), node_type='Medication')
    
    probability = round(float(row['Probability']), 3)
    UKG.add_edge(row['Disease'].lower(), row['Medication'].lower(), probability=probability)

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(modell_name)

def calculate_joint_probability(UKG, med1, med2):
    """
    Calculate the joint probability of taking both medications.
    """
    common_diseases = [disease for disease in UKG.predecessors(med1) if UKG.has_edge(disease, med2)]
    
    if not common_diseases:
        return 0.0
    
    total_prob = 0
    for disease in common_diseases:
        p_med1_given_disease = UKG[disease][med1]['probability']
        p_med2_given_disease = UKG[disease][med2]['probability']
        total_prob += p_med1_given_disease * p_med2_given_disease
    
    joint_probability = total_prob / len(common_diseases)
    
    return joint_probability

def calculate_conditional_probability(UKG, disease, med1, med2):
    """
    Calculate the conditional probability of taking both medications given the disease.
    """
    if not UKG.has_edge(disease, med1) or not UKG.has_edge(disease, med2):
        return 0.0
    
    p_med1_given_disease = UKG[disease][med1]['probability']
    p_med2_given_disease = UKG[disease][med2]['probability']
    
    joint_prob = p_med1_given_disease * p_med2_given_disease
    return joint_prob

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def query_model(disease, medication):
    def model(disease, medication):
        alpha = torch.tensor([UKG[disease][medication]['probability']])
        p = pyro.sample("p", dist.Beta(alpha[0], 1))
        return p

    mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model), num_samples=1000, warmup_steps=200)
    mcmc.run(disease, medication)
    posterior = mcmc.get_samples()["p"]
    probability = posterior.mean().item()
    return probability

def query_ukg_by_relationship(entity, relationship, include_probabilities=True):
    results = set()
    entity = entity.lower()
    if relationship == 'Medication':
        if entity in UKG:
            for neighbor in UKG.neighbors(entity):
                if include_probabilities:
                    results.add((neighbor, UKG[entity][neighbor]['probability']))
                else:
                    results.add(neighbor)
    elif relationship == 'Disease':
        for source, target in UKG.edges:
            if target == entity:
                if include_probabilities:
                    results.add((source, UKG[source][entity]['probability']))
                else:
                    results.add(source)
    return list(results)

def get_information(query):
    query = query.lower()
    if 'medication for' in query:
        entity = query.split('medication for')[1].strip()
        kg_response = query_ukg_by_relationship(entity, 'Medication', include_probabilities=False)
        if kg_response:
            response_lines = [f"Medications for '{entity}' are:"]
            for neighbor in kg_response:
                response_lines.append(f"- {neighbor}")
            response = "\n".join(response_lines)
        else:
            response = generate_text(f"Medication for {entity}")
    elif 'disease treated by' in query:
        entity = query.split('disease treated by')[1].strip()
        kg_response = query_ukg_by_relationship(entity, 'Disease', include_probabilities=False)
        if kg_response:
            response_lines = [f"Diseases treated by '{entity}' are:"]
            for neighbor in kg_response:
                response_lines.append(f"- {neighbor}")
            response = "\n".join(response_lines)
        else:
            response = generate_text(f"Disease treated by {entity}")
    elif 'probability of' in query:
        parts = query.split('probability of')[1].strip().split('treating')
        disease = parts[0].strip()
        medication = parts[1].strip()
        if UKG.has_edge(disease, medication):
            probability = UKG[disease][medication]['probability']
            response = f"The probability of '{medication}' treating '{disease}' is {probability}."
        else:
            response = generate_text(f"Probability of {disease} treating {medication}")
    else:
        response = "Invalid query format. Please use one of the formats:\n- 'Medication for Disease'\n- 'Disease treated by Medication'\n- 'Probability of Disease treating Medication'"
    return response

def handle_query():
    query = query_entry.get("1.0", tk.END).strip()
    if query.lower() == 'exit':
        root.destroy()
        return
    
    response = get_information(query)
    output_text.configure(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Query: {query}\n{response}\n\n")
    output_text.configure(state=tk.DISABLED)

def calculate_probabilities():
    med1 = med1_entry.get().strip().lower()
    med2 = med2_entry.get().strip().lower()
    disease = disease_entry.get().strip().lower()

    if med1 not in UKG.nodes or med2 not in UKG.nodes:
        result_text = f"One or both medications '{med1}' and '{med2}' are not in the graph."
    elif disease not in UKG.nodes:
        result_text = f"The disease '{disease}' is not in the graph."
    else:
        joint_prob = calculate_joint_probability(UKG, med1, med2)
        conditional_prob = calculate_conditional_probability(UKG, disease, med1, med2)
        result_text = (f"The probability of taking both '{med1}' and '{med2}' is {joint_prob:.3f}\n"
                       f"The probability of taking both '{med1}' and '{med2}' given '{disease}' is {conditional_prob:.3f}")
    
    probability_output.configure(state=tk.NORMAL)
    probability_output.delete("1.0", tk.END)
    probability_output.insert(tk.END, result_text)
    probability_output.configure(state=tk.DISABLED)

# Set up GUI
root = tk.Tk()
root.title("Knowledge Graph Query System")
root.geometry("600x600") 

# Set background color
root.configure(bg='#f0f0f0')

# Query entry
query_label = tk.Label(root, text="Enter your query:", font=("Helvetica", 10, 'bold'), bg='#f0f0f0', fg='#333333')
query_label.pack(pady=7)

query_entry = tk.Text(root, height=2, width=70, font=("Helvetica", 12))
query_entry.pack(pady=5)

# Output display
output_label = tk.Label(root, text="Query Results:", font=("Helvetica", 10, 'bold'), bg='#f0f0f0', fg='#333333')
output_label.pack(pady=7)

output_text = scrolledtext.ScrolledText(root, height=8, width=70, font=("Helvetica", 12), bg='#ffffff', fg='#000000', wrap=tk.WORD)
output_text.pack(pady=5)

# Query button
query_button = tk.Button(root, text="Submit Query", command=handle_query, font=("Helvetica", 12, 'bold'), bg='#4CAF50', fg='#ffffff', relief=tk.RAISED)
query_button.pack(pady=7)

# Probability calculation section
probability_label = tk.Label(root, text="Calculate Probabilities", font=("Helvetica", 10, 'bold'), bg='#f0f0f0', fg='#333333')
probability_label.pack(pady=10)

med1_label = tk.Label(root, text="Enter the first medication:", font=("Helvetica", 10), bg='#f0f0f0', fg='#333333')
med1_label.pack(pady=5)

med1_entry = tk.Entry(root, font=("Helvetica", 12))
med1_entry.pack(pady=5)

med2_label = tk.Label(root, text="Enter the second medication:", font=("Helvetica", 10), bg='#f0f0f0', fg='#333333')
med2_label.pack(pady=5)

med2_entry = tk.Entry(root, font=("Helvetica", 12))
med2_entry.pack(pady=5)

disease_label = tk.Label(root, text="Enter the disease for conditional probability:", font=("Helvetica", 10), bg='#f0f0f0', fg='#333333')
disease_label.pack(pady=5)

disease_entry = tk.Entry(root, font=("Helvetica", 12))
disease_entry.pack(pady=5)

# Probability calculation button
probability_button = tk.Button(root, text="Calculate Probabilities", command=calculate_probabilities, font=("Helvetica", 12, 'bold'), bg='#2196F3', fg='#ffffff', relief=tk.RAISED)
probability_button.pack(pady=7)

# Probability output display
probability_output_label = tk.Label(root, text="Probability Results:", font=("Helvetica", 10, 'bold'), bg='#f0f0f0', fg='#333333')
probability_output_label.pack(pady=7)

probability_output = scrolledtext.ScrolledText(root, height=3, width=80, font=("Helvetica", 10), bg='#ffffff', fg='#000000', wrap=tk.WORD)
probability_output.pack(pady=5)

# Instructions
instructions = """
Enter your query in one of the following formats:
- 'Medication for Disease'
- 'Disease treated by Medication'
- 'Probability of Disease treating Medication'
Type 'exit' to quit the program.
"""
instructions_label = tk.Label(root, text=instructions, font=("Helvetica", 12), bg='#f0f0f0', fg='#333333')
instructions_label.pack(pady=10)

# Run the main loop
root.mainloop()
