from src.report_gen import judge_feature

th = {"QTc": {"mild_high": 500}}

print(judge_feature("QTc", 450, th, sex="男"))
print(judge_feature("QTc", 450, th, sex="女"))
print(judge_feature("QTc", 450, th, sex=""))