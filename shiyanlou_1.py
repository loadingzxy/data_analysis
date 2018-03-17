import json

def analysis(file, user_id):
    times = 0
    minutes = 0

    json_str = open(file, "rb")
    data = json.load(json_str)

    for i in data:
        if user_id == i['user_id']:
            times = times + 1
            minutes = i['minutes'] + minutes
    print(times, minutes)


    return times, minutes

if __name__ == '__main__':
    path = 'user_study.json'
    uid = 199071
    analysis(path,uid)
