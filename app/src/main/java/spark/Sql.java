package spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Sql {

    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName("sr-app").setMaster("local");
        SparkSession spark = SparkSession
                                .builder()
                                .appName("sr-SQL")
                                .config(conf)
                                .getOrCreate();                            
        
        Dataset<Row> df = spark.read().json("src/main/resources/people.json");
        df.show();
        df.where("age>20");
    }
    
    
}



