#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Title     : TODO
# @Objective : TODO
# @Time      : 2018/2/27 10:34
# @Author    : hechuan24
# @Site      :
# @File      : sp_test_mysql.py
# @Software  : PyCharm
import sys
import datetime

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import HiveContext
from pyspark.sql import Row
from pyspark.sql.types import *

reload(sys)
sys.setdefaultencoding("utf-8")


# 读
def read_mysql():
    # 第一种方式
    df1 = hiveCtx.read.jdbc(url="jdbc:mysql://url?characterEncoding=utf-8",
                            table="table_name", properties=properties)
    df.show()

    # 第二种方式
    # df2 = hiveCtx.read.format("jdbc") \
    #  .option("url", "jdbc:mysql://url?characterEncoding=utf-8") \
    #  .option("dbtable", "table_name") \
    #  .option("user", "userName") \
    #  .option("password", "password") \
    #  .option("driver", "com.mysql.jdbc.Driver") \
    #  .load()
    # df2.show()


# 写
def write_mysql():
    # 读取hive表中数据 输出到mysql表中
    sqlDF = hiveCtx.sql("""
    select * from app.shop_xiala_shop_info limit 100
    """)

    ## mode : * ``append``: Append contents of this :class:`DataFrame` to existing data.
    #         * ``overwrite``: Overwrite existing data.
    #         * ``ignore``: Silently ignore this operation if data already exists.
    #         * ``error`` (default case): Throw an exception if data already exists.

    # 第一种方式
    sqlDF.write.jdbc(url="jdbc:mysql://url?characterEncoding=utf-8",
                     table="table_name", mode="overwrite", properties=properties)

    # 第二种方式
    # sqlDF.write \
    #     .format("jdbc") \
    #     .option("url", "jdbc:mysql://10.187.209.213:3306/search_misc?characterEncoding=utf-8") \
    #     .option("driver", "com.mysql.jdbc.Driver") \
    #     .option("dbtable", "table_name") \
    #     .option("user", "userName") \
    #     .option("password", "password") \
    #     .save(mode="overwrite")


if __name__ == "__main__":
    # 传入标准十位日期 2017-01-01
    dt = sys.argv[1]
    dt_str = datetime.datetime.strptime(dt, "%Y-%m-%d").date() if len(dt) == 10 \
        else datetime.datetime.strptime(dt, "%Y%m%d").date()
    dt_int = dt_str.strftime("%Y%m%d")
    dt_str_25 = (dt_str - datetime.timedelta(days=int(25))).strftime("%Y-%m-%d")

    conf = SparkConf()
    conf.set("spark.sql.codegen", "true")
    conf.set("spark.sql.inMemoryColumnarStorage.compressed", "true")
    conf.set("spark.sql.autoBroadcastJoinThreshold", "200000")
    sc = SparkContext(conf=conf, appName='app_name')
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf("hive.auto.convert.join", "true")
    hiveCtx.setConf("spark.sql.shuffle.partitions", "1000")
    hiveCtx.setConf("spark.default.parallelism", "1000")
    hiveCtx.setConf("mapreduce.job.reduces", "1000")
    hiveCtx.setConf("spark.shuffle.consolidateFiles", "true")
    hiveCtx.setConf("spark.shuffle.compress", "true")
    hiveCtx.sql("use app")

    # mysql 用户名及密码 以及mysql连接驱动全类名
    properties = {"user": "userName", "password": "password", "driver": "com.mysql.jdbc.Driver"}

    read_mysql()
    write_mysql()
    print("work done!")