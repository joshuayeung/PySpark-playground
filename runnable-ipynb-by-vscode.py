# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
file_location = "/FileStore/tables/game_skater_stats.csv"


#%%
df = spark.read.format("csv").option("inferSchema", 
                                     True).option("header", 
                                                  True).load(file_location)


#%%
display(df)


#%%
df.write.save('/FileStore/parquet/game_skater_stats',
             format='parquet')


#%%
df = spark.read.load("/FileStore/parquet/game_skater_stats")


#%%
display(df)


#%%
df = spark.read.load("s3a://my_bucket/game_skater_stats/*.parquet")


#%%
# DBFS (Parquet)
df.write.save('/FileStore/parquet/game_stats', format='parquet')


#%%
# S3 (Parquet)
df.write.parquet("s3a://my_bucket/game_stats", mode="overwrite")


#%%
# DBFS (CSV)
df.write.save('/FileStore/parquet/game_stats.csv', format='csv')


#%%
# S3 (CSV)
df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(
"s3a://my_bucket/game_stats.csv")


#%%
# Performing operations on Spark dataframes is via Spark SQL
df.createOrReplaceTempView("stats")

display(spark.sql("""
select player_id, sum(1) as games, sum(goals) as goals
from stats
group by 1
order by 3 desc
limit 5
"""))


#%%
display(spark.sql("""
select player_id, sum(1) as games, sum(goals) as goals
from stats
group by player_id
order by goals desc
limit 5
"""))


#%%
top_players = spark.sql("""
select player_id, sum(1) as games, sum(goals) as goals
from stats
group by 1
order by 3 desc
limit 5
""")


#%%
# player names
file_location = "/FileStore/tables/player_info.csv"
names = spark.read.format("CSV").option("inferSchema", True).option("header", True).load(file_location)


#%%
top_players.createOrReplaceTempView("top_players")
names.createOrReplaceTempView("names")


#%%
display(spark.sql("""
select p.player_id, goals, firstName, lastName
from top_players p
join names n
on p.player_id = n.player_id
order by 2 desc
"""))


#%%
display(spark.sql("""
  select cast(substring(game_id, 1, 4) || '-' 
    || substring(game_id, 5, 2) || '-01' as Date) as month
    , sum(goals)/count(distinct game_id) as goals_per_goal
  from stats
  group by 1
  order by 1
"""))


#%%
display(spark.sql("""
  select cast(goals/shots * 50 as int)/50.0 as Goals_per_shot
      ,sum(1) as Players 
  from (
    select player_id, sum(shots) as shots, sum(goals) as goals
    from stats
    group by 1
    having goals >= 5
  )  
  group by 1
  order by 1
"""))


#%%
# MLlib imports
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


#%%
# Create a vector representation for features
assembler = VectorAssembler(inputCols=['shots', 'hits', 'assists',
                                      'penaltyMinutes', 'timeOnIce', 'takeaways'],
                           outputCol="features")
train_df = assembler.transform(df)


#%%
# Fit a linear regression model


