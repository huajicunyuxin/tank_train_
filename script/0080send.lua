-- 定义要监视的RAM地址
local ram_address = 0x0080 -- 举例，这里使用0x00C0作为要监视的地址

-- 创建一个函数来读取并打印RAM地址的值
local function monitor_ram()
    local value = memory.readbyte(ram_address)
    print(string.format("add0x%04X : %d", ram_address, value))
    -- 创建并打开一个文件用于写入
local file = io.open("hello_world.txt", "w")

-- 检查文件是否成功打开
if file then
    -- 写入字符串"Hello World"
    file:write(value)
    -- 关闭文件
    file:close()
    -- 输出信息到控制台
    print("success")
else
    print("fail")
end
end 

-- 主循环，不断调用monitor_ram函数来监视RAM
while true do
    monitor_ram()
    emu.frameadvance() -- 这会使模拟器前进一帧
end


-- -- 创建并打开一个文件用于写入
-- local file = io.open("hello_world.txt", "w")

-- -- 检查文件是否成功打开
-- if file then
--     -- 写入字符串"Hello World"
--     file:write("Hello World")
--     -- 关闭文件
--     file:close()
--     -- 输出信息到控制台
--     print("success")
-- else
--     print("fail")
-- end