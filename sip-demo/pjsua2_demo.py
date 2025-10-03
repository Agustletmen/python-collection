# pip install pjsua2

import pjsua2 as pj
import time

# 自定义呼叫回调类（处理呼叫状态变化）
class Call(pj.Call):
    def __init__(self, acc, call_id=pj.PJSUA_INVALID_ID):
        pj.Call.__init__(self, acc, call_id)

    # 处理呼叫状态变化
    def onCallState(self, prm):
        ci = self.getInfo()
        print(f"Call state: {ci.stateText}, reason: {ci.lastReason}")
        # 当呼叫被接听时
        if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            print("Call connected!")
        # 当呼叫结束时
        elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            print("Call disconnected.")

    # 处理媒体状态变化
    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                # 激活音频流
                aud_dev_manager = pj.Endpoint.instance().audDevManager()
                aud_dev_manager.startTransmit(mi.audioFrame, mi.audioFrame)
                print("Audio stream active.")

# 主函数
def main():
    # 初始化端点
    ep = pj.Endpoint()
    ep.libCreate()

    # 配置端点
    ep_cfg = pj.EpConfig()
    ep_cfg.logConfig.level = 3  # 日志级别（0-5）
    ep_cfg.logConfig.consoleLevel = 3
    ep_cfg.logConfig.filename = "pjsua2_log.txt"  # 日志文件

    # 初始化库
    ep.libInit(ep_cfg)

    # 配置传输（UDP）
    sipTpConfig = pj.TransportConfig()
    sipTpConfig.port = 5060  # 本地 SIP 端口
    ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, sipTpConfig)

    # 启动库（开始处理事件）
    ep.libStart()
    print("Library started.")

    # 配置 SIP 账户
    acc_cfg = pj.AccountConfig()
    acc_cfg.idUri = "sip:your_username@your_sip_server"  # 账户 URI
    acc_cfg.regConfig.registrarUri = "sip:your_sip_server"  # 注册服务器

    # 认证信息
    cred = pj.AuthCredInfo("digest", "*", "your_username", 0, "your_password")
    acc_cfg.sipConfig.authCreds.append(cred)

    # 创建并注册账户
    acc = pj.Account()
    acc.create(acc_cfg)
    print("Registering account...")
    time.sleep(2)  # 等待注册完成

    # 发起呼叫
    call_uri = "sip:destination_user@your_sip_server"  # 目标 SIP URI
    call = Call(acc)
    call_param = pj.CallOpParam()
    try:
        call.makeCall(call_uri, call_param)
        print(f"Calling {call_uri}...")
    except Exception as e:
        print(f"Call failed: {e}")
        return

    # 保持程序运行（等待呼叫事件）
    print("Press Enter to hang up...")
    input()

    # 挂断呼叫
    hangup_param = pj.CallOpParam()
    call.hangup(hangup_param)
    time.sleep(1)

    # 清理资源
    ep.libDestroy()
    print("Library destroyed.")

if __name__ == "__main__":
    main()