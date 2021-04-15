import os, json

def append_new_entry(json_file_path: str, content: dict):
    with open (json_file_path, mode="r+") as file:
        file.seek(os.stat(json_file_path).st_size -1)
        file.write("{ }" + ",{}]".format(json.dumps(content)))

# # Another method without os library
# for i in range(5):
#     with open (filepath, mode="r+") as file:
#         file.seek(0,2)
#         position = file.tell() -1
#         print(position)
#         file.seek(position)
#         file.write( ",{}]".format(json.dumps(dictionary)))
