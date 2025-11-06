# 🧠 Utilising Large Language Models for Generating Effective Counter Arguments to Anti-Vaccine Tweets

This repository accompanies the paper:  
**“Utilising Large Language Models for Generating Effective Counter Arguments to Anti-Vaccine Tweets”**  
by *Utsav Dhanuka et al.*

📄 **Paper (arXiv):** [https://arxiv.org/abs/2510.16359](https://arxiv.org/abs/2510.16359)

---

## 🔍 Overview  

This project investigates the ability of **Large Language Models (LLMs)** to generate effective counter-arguments against anti-vaccine misinformation on social media platforms such as Twitter (X).  
We perform both **quantitative** and **qualitative** analyses comparing **human-written** and **LLM-generated** counter-arguments.  

Our findings demonstrate that with carefully designed prompts and fine-tuning, LLMs can produce **persuasive, factual, and coherent** counter-arguments that rival or even surpass human-written responses in certain aspects.  

---

## 🚀 Key Contributions  

- **LLM vs. Human Evaluation:** A systematic comparison of human and model-generated counter-arguments using both automatic metrics (ROUGE, BERTScore) and human judgment (coverage, clarity, persuasiveness, factualness).  
- **CNTR-VAX Dataset:** A novel dataset containing anti-vaccine tweets and their corresponding counter-arguments — both **label-aware** and **label-free** versions — generated using GPT-4o-mini.  
- **Fine-Tuning Experiments:** Extensive experiments using compact LLMs (Phi-3 Mini, LLaMA-3.3B, Gemma 2-bit) to explore label-awareness and prompt design strategies.  
- **Label-Aware Prompting:** Demonstrates that incorporating descriptive misinformation labels enhances factual grounding and argumentative quality.  
- **Human Survey Evaluation:** A structured human evaluation assessing the persuasiveness, factualness, and clarity of generated responses.

---


---

## 📈 Evaluation Metrics  

| **Category** | **Metrics** |
|---------------|-------------|
| Automatic Evaluation | ROUGE-1, ROUGE-2, ROUGE-L, BERTScore (Precision, Recall, F1) |
| Human Evaluation | Coverage, Clarity, Factualness, Persuasiveness |

---

## 🧪 Experimental Setup  

- **Models Used:** GPT-4o-mini, Phi-3 Mini, LLaMA-3.3B, Gemma 2-bit, and others.  
- **Training Data:** CNTR-VAX dataset — 2,000 training and 990 evaluation samples.  
- **Evaluation Scenarios:**  
  1. Prompts without label descriptions  
  2. Prompts with ground-truth label descriptions  
  3. Prompts with predicted labels (two-step pipeline)  
  4. Chain-of-Thought reasoning with larger LLMs  

---

## 💡 Insights  

- Label-aware prompting significantly improves the **factual grounding** and **persuasiveness** of generated counter-arguments.  
- Even small fine-tuned models perform competitively when trained with targeted data and optimized prompts.  
- Human evaluators rated label-aware responses as more **informative**, **nuanced**, and **convincing**.  

---

## 🧠 Citation  

If you use this work, please cite:  

```bibtex
@misc{dhanuka2025utilisinglargelanguagemodels,
      title={Utilising Large Language Models for Generating Effective Counter Arguments to Anti-Vaccine Tweets}, 
      author={Utsav Dhanuka and Soham Poddar and Saptarshi Ghosh},
      year={2025},
      eprint={2510.16359},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.16359}, 
}



