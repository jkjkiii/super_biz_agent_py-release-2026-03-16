# MySQL 连接池耗尽告警处理方案

## 告警名称
- **告警名**: `MySQLConnectionPoolExhausted`
- **告警级别**: 严重
- **触发条件**: 数据库连接池使用率超过 90% 持续 5 分钟

## 问题描述
连接池耗尽会导致新请求无法获取数据库连接，业务操作失败。

## 排查步骤

### 步骤1: 查看当前连接数
```sql
SHOW PROCESSLIST;
SELECT COUNT(*) FROM information_schema.processlist;
步骤2: 分析连接状态
sql
SELECT state, COUNT(*) FROM information_schema.processlist GROUP BY state;
步骤3: 查询应用日志
工具: query_logs
参数要求:

地域: ap-guangzhou

日志主题: application-logs

查询条件: DataSource.getConnection() timeout

常见原因分析
原因1: 慢 SQL 导致连接长时间占用
特征:

连接池活跃连接数持续高位

存在执行时间很长的 SQL

处理方案:

查询 information_schema.processlist 找出慢连接

kill 慢查询会话

优化 SQL 或增加索引

原因2: 连接未正确释放
特征:

应用代码中未关闭 Connection 或 ResultSet

连接泄漏

处理方案:

检查代码，确保 try-with-resources 或 finally 块释放连接

配置连接池的 leakDetectionThreshold

重启应用临时释放所有连接

原因3: 连接池配置过小
特征:

业务量增长，但 maxActive 未调整

处理方案:

适当增加 maxActive 和 maxIdle

调整 maxWait 超时时间

紧急处理措施
临时增大连接池大小

重启应用释放所有连接

限流减少并发请求

验证步骤
确认连接池使用率降到正常水平（<70%）

检查业务请求成功率

观察数据库负载