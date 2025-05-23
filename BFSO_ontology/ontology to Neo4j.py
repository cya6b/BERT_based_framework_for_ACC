from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import json
# 连接图库                            初始化账号密码都是neo4j
graph = Graph('http://localhost:7474', auth=('neo4j', '12345678'))

matcher = NodeMatcher(graph)

file = open("./data/SpanR.jsonl", 'r', encoding='utf-8')
data = []

for line in file.readlines():
    dic = json.loads(line)
    data.append(dic)

for item in data:
    entities = item["entities"]
    relations = item["relations"]
    text = item["text"]

    for relation in relations:
        from_entity_id = relation["from_id"]
        to_entity_id = relation["to_id"]
        relation_type = relation["type"]

        from_entity = next(entity for entity in entities if entity["id"] == from_entity_id)
        to_entity = next(entity for entity in entities if entity["id"] == to_entity_id)

        from_entity_text = text[from_entity["start_offset"]:from_entity["end_offset"]]
        to_entity_text = text[to_entity["start_offset"]:to_entity["end_offset"]]

        from_node = matcher.match(from_entity["label"]).where("_.name=" + "'" + from_entity_text + "'").first()
        if not from_node:
            from_node = Node(from_entity["label"], name=from_entity_text)
            graph.create(from_node)

        to_node = matcher.match(to_entity["label"]).where("_.name=" + "'" + to_entity_text + "'").first()
        if not to_node:
            to_node = Node(to_entity["label"],name=to_entity_text)
            graph.create(to_node)


        relationship = Relationship(from_node, relation_type, to_node)
        graph.create(relationship)

    # # 头实体
    # head = Node("subject", name=from_entity_text)
    # # 尾实体
    # tail = Node("object", name=to_entity_text)
    # # entity = Relationship(head, relation_type, tail)
    # graph.create(tail)
    #
    #
    # # 匹配查找图库中节点
    # nodelist = list(matcher.match("object").where("_.name=" + "'" + to_entity_text + "'").first())
    # print(len(nodelist))
    # if len(nodelist) > 0:
    #     # 表示节点存在，不需创建新的节点
    #     already_header = nodelist[0]
    #     # 可以直接添加关系
    #     entity = Relationship(head, relation_type, already_header)
    #     graph.create(entity)
    # else:
    #     # 表示图库中没有存在当前要写入的节点
    #     entity = Relationship(head, relation_type, tail)
    #     # 创建关系
    #     graph.create(entity)

    #print(from_entity_name)
    # print("From Entity:", from_entity_text)
    # print("Relation Type:", relation_type)
    # print("To Entity:", to_entity_text)
    # print("---")





#triples=[]
#for item in data:
# graph.delete_all()





