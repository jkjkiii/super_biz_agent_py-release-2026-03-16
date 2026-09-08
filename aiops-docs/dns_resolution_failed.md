# DNS 解析失败告警处理方案

## 告警名称
- **告警名**: `DNSResolutionFailed`
- **告警级别**: 严重
- **触发条件**: DNS 解析失败率超过 2% 持续 3 分钟

## 问题描述
DNS 解析失败会导致服务发现异常，应用无法调用依赖服务。

## 排查步骤

### 步骤1: 验证 DNS 解析
```bash
dig <domain> @<dns_server>
nslookup <domain>
步骤2: 查看 DNS 缓存
bash
cat /etc/resolv.conf
systemctl status nscd
步骤3: 查询 DNS 日志
工具: query_logs
参数要求:

地域: ap-guangzhou

日志主题: system-logs

查询条件: DNS resolution error

常见原因分析
原因1: DNS 服务器不可达
特征:

connection timed out

DNS 服务器 IP 不可达

处理方案:

检查 DNS 服务器状态

切换到备用 DNS 服务器

检查防火墙规则

原因2: 域名记录变更未同步
特征:

新域名解析失败

TTL 未过期

处理方案:

手动刷新 DNS 缓存

等待 TTL 过期

检查 DNS 记录配置

原因3: /etc/hosts 配置错误
特征:

本地 hosts 文件包含错误映射

处理方案:

检查并修正 /etc/hosts

重启网络服务

紧急处理措施
临时修改 /etc/hosts 添加静态映射

重启应用容器刷新 DNS

切换 DNS 服务器

验证步骤
确认域名解析成功

测试依赖服务调用

监控 DNS 解析延迟