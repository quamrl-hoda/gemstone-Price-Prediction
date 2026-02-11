
try:
    with open('logs/running_logs.log', 'r') as f:
        content = f.read()
        print(content[-2000:])
except Exception as e:
    print(e)
