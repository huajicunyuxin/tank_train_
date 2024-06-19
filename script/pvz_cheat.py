import time


def num_enemy():
    # 文件路径
    file_path = 'E:\\智能系统课程设计\\reinforce_test_1 (2)\\script\\hello_world.txt'

    try:
        # 打开并读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 尝试将内容转换为整数
            num = int(content)
            #print(num)
            return num
    except ValueError:
        print("无法将文件内容转换为整数")
        return -1
    except KeyboardInterrupt:
        print("程序已停止")
        return -1
    