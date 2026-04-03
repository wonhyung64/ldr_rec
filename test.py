import inspect
file_dir = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
file_name = file_dir.split("/")[-1]
if file_name.split(".")[-1] == "py":
    print(1)
else:
    print(0)