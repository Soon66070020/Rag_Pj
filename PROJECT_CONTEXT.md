แน่นอนครับ ผมได้เขียน **Product Requirement Document (PRD)** ฉบับสมบูรณ์ใหม่ทั้งหมด โดยขยายความส่วน **Functional Requirements (FR)** ให้ครอบคลุมการทำงานของระบบตั้งแต่ต้นจนจบ (End-to-End) ทั้งฝั่ง Admin, User และ System Process เพื่อให้เหมาะกับการทำวิทยานิพนธ์ระดับปริญญาตรีที่ต้องลงรายละเอียดเชิงลึกครับ

```

#### 7. Technical Specifications (Updated)

เพิ่มตัวอย่าง **System Prompt** เพื่อให้ Developer นำไปใช้:

**System Prompt Template:**

```

---

# Product Requirement Document (PRD)

**Project Name:** Post-Oral Surgery Intelligent Assessment & Advisory System (RAG-based)
**Document Version:** 2.0 (Comprehensive Release)
**Date:** 2026-02-02
**Status:** Ready for Thesis Proposal / Development

---

## 1. บทนำ (Introduction)

### 1.1 ที่มาและความสำคัญ (Background)

ผู้ป่วยหลังการผ่าตัดในช่องปาก (เช่น ผ่าฟันคุด, ศัลยกรรมขากรรไกร) มักประสบปัญหาความกังวลเกี่ยวกับอาการหลังผ่าตัด การค้นหาข้อมูลทั่วไปอาจได้คำแนะนำที่ไม่ถูกต้อง (Misinformation) หรือไม่ตรงกับแนวทางการรักษาของแพทย์เจ้าของไข้
ระบบนี้จึงถูกพัฒนาขึ้นเพื่อเป็น **AI-based Assistant** ที่สามารถตอบคำถามและประเมินอาการโดยใช้เทคนิค **Retrieval-Augmented Generation (RAG)** ขั้นสูง ที่เน้นความแม่นยำ (High Accuracy) โดยจำกัดขอบเขตคำตอบให้อยู่ภายใต้เอกสารทางการแพทย์ที่กำหนด (Closed-Domain) เพื่อลดปัญหา Hallucination

### 1.2 วัตถุประสงค์ (Objectives)

1. เพื่อพัฒนาระบบตอบคำถามอัตโนมัติสำหรับผู้ป่วยศัลยกรรมช่องปากที่มีความแม่นยำสูง
2. เพื่อเปรียบเทียบประสิทธิภาพของการใช้เทคนิค Hybrid Retrieval (Dense + Sparse) ร่วมกับ Reranking
3. เพื่อประเมินผลระบบด้วยเกณฑ์มาตรฐานทั้งเชิงปริมาณ (Metrics) และเชิงคุณภาพ (Human Eval)

---

## 2. กลุ่มผู้ใช้งาน (User Personas)

* **End-User (Patient):** ผู้ป่วยที่ต้องการคำปรึกษาเร่งด่วน หรือคำแนะนำการปฏิบัติตัวหลังผ่าตัด
* **Administrator (Researcher/Dentist):** ผู้ที่มีหน้าที่นำเข้าคู่มือการรักษา (Knowledge Base) และตรวจสอบประสิทธิภาพการตอบของระบบ

---

## 3. สถาปัตยกรรมระบบ (System Architecture)

* **Large Language Model (LLM):** **DeepSeek API (Model: deepseek-chat / v3.2)**
* **Embedding Model:** **BGE-M3** (รองรับ Multi-linguality และสร้างได้ทั้ง Dense & Sparse Vectors)
* **Vector Database:** **Weaviate** (รองรับ Hybrid Search และ Metadata Filtering)
* **Reranker Model:** **BGE-Reranker-v2-m3**
* **Orchestration Framework:** LangChain
* **Python:** Python 3.10+
* **Docstring:** เขียน Docstring แบบ Google Style

---

## 4. ความต้องการเชิงหน้าที่ (Functional Requirements)

ได้ทำการขยายรายละเอียดแบ่งตามโมดูลการทำงานจริง เพื่อให้เห็นภาพรวมของระบบที่สมบูรณ์

### Module 1: Knowledge Base Management (สำหรับ Admin)

* **FR-01 Document Ingestion:**
* ระบบต้องรองรับการอัปโหลดไฟล์เอกสารรูปแบบ PDF, TXT, และ Markdown
* ระบบต้องทำการแปลงไฟล์ (Parsing) ให้อยู่ในรูป Text format ที่สะอาด


* **FR-02 Advanced Chunking Strategy:**
* ระบบต้องใช้เทคนิค **Semantic Chunking** ในการตัดแบ่งข้อความ เพื่อให้แต่ละท่อน (Chunk) มีใจความสมบูรณ์
* ระบบต้องรองรับการกำหนด Chunk Size และ Overlap ที่เหมาะสม (เช่น 512 tokens / 50 overlap)


* **FR-03 Automated Metadata Extraction:**
* ระบบต้องแยกแยะและบันทึก Metadata อัตโนมัติหรือกึ่งอัตโนมัติ ได้แก่:
* `Category` (เช่น Post-op Care, Medication, Emergency)
* `Source Filename` (ชื่อไฟล์อ้างอิง)
* `Page Number` (เลขหน้า)




* **FR-04 Dual-Embedding Indexing:**
* ระบบต้องสร้าง Index 2 รูปแบบสำหรับทุก Chunk:
1. **Dense Vector:** สำหรับจับความหมายโดยรวม (Semantic)
2. **Learned Sparse Vector (from BGE-M3):** สำหรับจับ Keyword สำคัญ (Lexical) แทนการใช้ BM25





### Module 2: Retrieval Engine (Core System Logic)

* **FR-05 Query Pre-processing (HyDE):**
* เมื่อได้รับคำถาม ระบบต้องทำการสร้าง "คำตอบสมมติ" (Hypothetical Document Embedding) หรือปรับปรุงรูปประโยค (Query Rewriting) เพื่อเพิ่ม Context ให้กับคำถามสั้นๆ


* **FR-06 Metadata Pre-filtering:**
* ระบบต้องสามารถกรองข้อมูลก่อนค้นหา (Pre-filtering) โดยดูจาก Intent ของคำถาม (เช่น ถ้าถามเรื่อง "ยา" ให้ตัด Chunk หมวด "อาหาร" ทิ้งไปก่อน)


* **FR-07 Hybrid Search Execution:**
* ระบบต้องค้นหาข้อมูลจาก Weaviate โดยใช้ **Alpha Blending** ผสมคะแนนระหว่าง Dense Score และ Sparse Score เพื่อให้ได้รายการเอกสารตั้งต้น (Candidate Set) จำนวน 20 รายการ (Top-K = 20)


* **FR-08 Context Reranking:**
* ระบบต้องนำ Candidate Set ทั้ง 20 รายการ ส่งเข้าโมเดล **BGE-Reranker** เพื่อคำนวณคะแนนความเกี่ยวข้องใหม่ (Relevance Score) และคัดเหลือเพียง 5 รายการที่ดีที่สุด (Top-N = 5)



### Module 3: Response Generation (User Interface Flow)

* **FR-09 Prompt Construction:**
* ระบบต้องนำ Context (5 รายการที่คัดมา) + คำถามผู้ใช้ + System Instruction มารวมกันเป็น Prompt
* System Instruction ต้องกำกับให้ AI ตอบในบทบาท "ผู้ช่วยทันตแพทย์" และ "ห้ามตอบนอกเหนือจากข้อมูลที่ให้"


* **FR-10 LLM Integration (DeepSeek):**
* ระบบต้องส่ง Prompt ไปยัง **DeepSeek API (v3.2)** และรับคำตอบกลับมาแสดงผล


* **FR-11 Citation & Reference (Strict Enforcement):**
* ทุกข้อความที่ระบบตอบกลับ (Response) **ต้อง** ระบุแหล่งที่มาข้อมูลในรูปแบบที่กำหนดเสมอ ห้ามละเว้น
* **Format:** เนื้อหาคำตอบ... `[Source: ชื่อไฟล์.pdf, Page: XX]`
* หากข้อมูลมาจากหลายหน้า ให้ระบุแยกกัน เช่น `[Source: FileA.pdf, Page: 10], [Source: FileB.pdf, Page: 3]`



### Module 4: Evaluation & Logging (สำหรับการวัดผล Thesis)

* **FR-12 Interaction Logging:**
* ระบบต้องบันทึกคู่ "คำถาม-คำตอบ" และ "Context ที่ดึงมาได้" ลงใน Log file หรือ Database เพื่อนำไปประเมินผล


* **FR-13 Automated Evaluation Pipeline:**
* ระบบต้องมี Script หรือ Module สำหรับรันการทดสอบตาม Metrics ที่กำหนด (BERTScore, ROUGE, Precision@K) เทียบกับ Golden Dataset



---

## 5. ความต้องการเชิงคุณภาพ (Non-Functional Requirements)

* **NFR-01 Accuracy (Faithfulness):** คำตอบต้องมีความถูกต้องตามเอกสารอ้างอิง > 90% (วัดผลด้วย Human Eval & Ragas)
* **NFR-02 Security & Privacy:** ข้อมูลส่วนตัวของผู้ป่วย (PII) เช่น ชื่อ-นามสกุล ต้องถูก Masking ก่อนส่งไปยัง API ภายนอก
* **NFR-03 Latency:** กระบวนการประมวลผลทั้งหมดรวมถึง Evaluation Loop (ถ้ามี) ต้องใช้เวลา **ไม่เกิน 1 ชั่วโมง** (ตามข้อกำหนดโครงการ) แต่สำหรับการตอบโต้ปกติ (Interactive) ควรพยายามให้ต่ำที่สุดเท่าที่เป็นไปได้
* **NFR-04 Scalability:** Vector Database ต้องรองรับการขยายตัวของข้อมูลเอกสารได้ในอนาคตโดยไม่ต้องรื้อระบบใหม่

---

## 6. แผนการวัดผล (Evaluation Metrics Matrix)

อ้างอิงตามมาตรฐานที่กำหนด แบ่งเป็น 4 มิติหลักดังนี้:

### 6.1 Human Evaluation

**Evaluators:** Dentists
**Scale:** Likert 1-5 (1-2=Bad, 3=Neutral, 4-5=Good)

| Metric | Question Criteria |
| --- | --- |
| **Correctness** | Is the information provided in the answer scientifically accurate and based on medical evidence? Are there any factual mistakes or outdated information? |
| **Relevance** | Does the answer address the question asked? Is the response related to the topic or problem mentioned in the question? |
| **Helpfulness** | Does the answer provide practical advice or actionable insights? Does the response improve understanding or offer recommendations beneficial to the patient? |

### 6.2 LLM-as-a-Judge Evaluation

**Judge Model:** LLM (High Intelligence Model)
**Scale:** Likert 1-5

| Metric | Prompt Criteria |
| --- | --- |
| **Correctness** | Evaluate if the response is scientifically accurate based on provided context. Penalize hallucinations. |
| **Relevance** | Evaluate if the response directly answers the user's specific query. |
| **Helpfulness** | Evaluate if the response offers actionable and beneficial advice for a post-op patient. |

### 6.3 Generation-based Evaluation (System Response vs Golden Response)

| Metric | Objective |
| --- | --- |
| **BERTScore** | Measures semantic similarity using BERT embeddings (F1-score of Precision/Recall). |
| **BLEU** | Measures n-gram precision overlap (Exact wording match). |
| **METEOR** | Measures completeness considering synonyms, stemming, and paraphrasing. |
| **ROUGE-I** | Measures n-gram recall (Coverage of important information). |

### 6.4 Retrieval-based Evaluation (RAG Pipeline Quality)

| Metric | Objective |
| --- | --- |
| **Precision@K** | Proportion of relevant items in the top K retrieved results. |
| **Recall@K** | Proportion of *all* relevant items that were successfully retrieved in top K. |
| **MRR** | (Mean Reciprocal Rank) Average rank of the *first* relevant item (Higher is better). |
| **nDCG** | (Normalized Discounted Cumulative Gain) Measures ranking quality, giving more weight to relevant items at the top. |
| **Similarity Score** | Vector cosine similarity between Query and Retrieved Documents. |
| **Groundedness** | Score evaluating if the generated response is fully supported by retrieved docs. |
| **Completeness** | Score evaluating if retrieved docs cover all aspects of the query. |
| **Relevance Score** | Overall pertinence of retrieved documents to the query context. |

---

## 7. ข้อมูลทางเทคนิค (Technical Specifications)

### 7.1 DeepSeek API Configuration

* **Endpoint:** `https://api.deepseek.com/v1/chat/completions`
* **Model Name:** `deepseek-chat` (v3.2)
* **Parameters:**
* `temperature`: 0.1 (Strict & Precise for Medical context)
* `top_p`: 0.9
* `max_tokens`: 2048
* `System Prompt Template`: """You are an expert dental assistant AI. 
Your task is to answer patient questions based ONLY on the provided context.

Rules:
1. You must answer in Thai language, using a polite and professional tone.
2. Do not use outside knowledge. If the answer is not in the context, say "I don't have enough information."
3. CRITICAL: Every sentence or paragraph you generate MUST be supported by a citation.
4. CITATION FORMAT: You must strictly use this format at the end of the answer or relevant sentence: [Source: filename, Page: number].

Context:
{context_str}

User Question:
{query}"""



### 7.2 Weaviate Schema Design (Draft)

```json
{
  "class": "MedicalGuideline",
  "description": "Stores chunks of post-op dental surgery guidelines",
  "vectorizer": "none", // Managed externally by BGE-M3
  "properties": [
    { "name": "content", "dataType": ["text"] },
    { "name": "category", "dataType": ["text"], "indexFilterable": true },
    { "name": "source_file", "dataType": ["text"] },
    { "name": "page_number", "dataType": ["int"] }
  ]
}

```