import utils
query = ["雅生活服务的人工成本占营业成本的比例是多少"]

context_list = []
with open("./dataset/financial_research_reports/yanbao001.txt","r",encoding="UTF-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        context_list.append(line)

print("下面是经过文本查询的结果如下所示：")
# context_list是以行分割的yanbao001内容list
sim_results = utils.get_top_n_sim_text(query=query[0],documents=context_list)
print(sim_results)
print("----------------------------------------")

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",trust_remote_code=True).half().cuda()

print("由chatGLM根据文档的严格回答的结果如下：")
# 显示地告诉ChatGLM不可以凭借经验或以往的训练内容来回答问题
prompt = utils.strict_generate_prompt(query[0],sim_results)
response, _ = model.chat(tokenizer, prompt, history=[])
print(response)
print("----------------------------------------")

