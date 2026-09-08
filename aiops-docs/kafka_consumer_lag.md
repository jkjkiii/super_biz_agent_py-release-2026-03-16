# Kafka 消费延迟告警处理方案

## 告警名称
- **告警名**: `KafkaConsumerLag`
- **告警级别**: 警告
- **触发条件**: 消费组 Lag 超过 10000 条持续 10 分钟

## 问题描述
Kafka 消费延迟会导致数据积压，实时性降低，可能引发数据不一致。

## 排查步骤

### 步骤1: 查看消费组状态
```bash
kafka-consumer-groups --bootstrap-server <broker> --group <group> --describe
步骤2: 检查分区分配情况
bash
kafka-consumer-groups --bootstrap-server <broker> --group <group> --describe --members
步骤3: 查看消费者日志
工具: query_logs
参数要求:

地域: ap-guangzhou

日志主题: application-logs

查询条件: Consumer poll timeout OR rebalance

常见原因分析
原因1: 消费者处理能力不足
特征:

Lag 持续增长

CPU 或内存使用率高

处理方案:

增加消费者实例数量

优化消费逻辑，减少单条消息处理时间

增加 max.poll.records 批量拉取

原因2: 消费者 Rebalance 频繁
特征:

日志中有 Rebalance 事件

消费停滞

处理方案:

增加 session.timeout.ms 和 max.poll.interval.ms

避免消费者心跳超时

检查网络稳定性

原因3: 生产者发送过快
特征:

消息生产量突增

处理方案:

对生产者限流

增加分区数提高并行度

紧急处理措施
临时增加消费者实例

降低处理逻辑复杂度（如跳过非关键处理）

扩容 Kafka 分区

验证步骤
确认 Lag 开始下降

检查消息消费速率

观察业务指标是否正常