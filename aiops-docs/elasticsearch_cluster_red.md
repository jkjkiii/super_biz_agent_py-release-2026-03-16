# Elasticsearch 集群状态 Red 告警处理方案

## 告警名称
- **告警名**: `ElasticsearchClusterRed`
- **告警级别**: 紧急
- **触发条件**: 集群状态为 Red 持续 3 分钟

## 问题描述
集群 Red 表示部分主分片未分配，读写操作可能失败，索引不可用。

## 排查步骤

### 步骤1: 查看集群健康状态
```bash
curl -X GET "localhost:9200/_cluster/health?pretty"
步骤2: 查看未分配分片原因
bash
curl -X GET "localhost:9200/_cluster/allocation/explain?pretty"
步骤3: 查看节点状态
bash
curl -X GET "localhost:9200/_nodes/stats?pretty"
常见原因分析
原因1: 节点磁盘空间不足
特征:

日志中出现 disk watermark exceeded

节点磁盘使用率超过 85%

处理方案:

清理旧索引（删除或归档）

扩容磁盘

临时调整磁盘水位阈值

原因2: 节点宕机或网络分裂
特征:

部分节点不可达

集群节点数变化

处理方案:

重启宕机节点

检查网络连通性

手动分配分片

原因3: 分片大小过大
特征:

单个分片超过 50GB

处理方案:

重建索引，调整分片数

使用 _split 或 _shrink API

紧急处理措施
临时增加磁盘空间

强制分配未分配分片

增加节点数量

验证步骤
确认集群状态变为 Green 或 Yellow

测试索引读写

监控集群稳定性