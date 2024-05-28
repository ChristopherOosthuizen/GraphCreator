from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = T5ForConditionalGeneration.from_pretrained("iarfmoose/t5-base-question-generator")
def chunks_to_questions(chunks):
    result = []
    for chunk in chunks:
        input_ids = tokenizer(chunk, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        result.append(tokenizer.decode(outputs[0]))
    return result