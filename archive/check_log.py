import json 

with open("logs/good_all.json") as f:
    data = json.load(f)
    for i in data:
        if i != {}:
            prev_acc = -1
            for stage in i["Stages"]:
                if prev_acc == -1:
                    prev_acc = i["Stages"][stage]["Character Accuracy"]
                else:
                    if i["Stages"][stage]["Character Accuracy"] < prev_acc:
                        print(i["Image Path"])
                        print(prev_acc)
                        print(i["Stages"][stage]["Character Accuracy"])
                    prev_acc = i["Stages"][stage]["Character Accuracy"]
