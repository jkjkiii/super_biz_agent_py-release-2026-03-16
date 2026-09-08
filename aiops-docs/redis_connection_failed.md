# Redis 连接失败告警处理方案

## 告警名称
- **告警名**: `RedisConnectionFailed`
- **告警级别**: 严重
- **触发条件**: Redis 连接成功率低于 80% 持续 3 分钟

## 问题描述
Redis 连接失败会导致缓存不可用，增加数据库压力，可能引发雪崩效应。

## 排查步骤

### 步骤1: 获取当前时间
**工具**: `get_current_time`

### 步骤2: 查询 Redis 监控日志
**工具**: `query_logs`
**参数要求**:
- **地域**: `ap-guangzhou`
- **日志主题**: `redis-metrics`
- **时间范围**: 最近 15 分钟
- **查询条件**: `connection_error OR timeout`

### 步骤3: 验证 Redis 连通性
```bash
redis-cli -h <host> -p <port> -a <password> ping
常见原因分析
原因1: 连接池耗尽
特征:

应用日志中出现 JedisConnectionException

连接数达到 maxTotal 限制

处理方案:

增加 maxTotal 和 maxIdle 参数

检查是否存在连接未释放

重启应用释放连接池

原因2: 网络超时
特征:

日志中出现 SocketTimeoutException

Redis 响应缓慢

处理方案:

检查网络延迟和丢包

调整 timeout 和 soTimeout 参数

考虑使用内网地址

原因3: Redis 服务不可用
特征:

Redis 进程崩溃

内存不足导致 OOM

处理方案:

重启 Redis

检查内存使用，增加内存或设置 maxmemory

检查 Redis 日志定位根因

紧急处理措施
立即重启应用实例，重建连接池

临时切换到本地缓存（如果业务允许）

扩容 Redis 集群

验证步骤
确认连接成功率恢复正常（>95%）

检查应用缓存命中率

观察数据库负载是否下降