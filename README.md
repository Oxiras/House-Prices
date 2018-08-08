# House-Prices
Gradient Boosted Trees to predict house prices.

HOW TO RUN

Option 1: Requires maven to be installed on pc
In project directory execute following command: 
mvn package exec:java -Dexec.cleanupDaemonThreads=false -Dexec.mainClass=edu.utdallas.Program

Option 3: Eclipse
1. Import project as maven project
2. Right-click on class 'Program.scala'
3. Run As 'Scala Application'

Option 2: AWS
1. Save jar file in a S3 bucket
2. Save train and test csvs to same S3 bucket
2. Copy jar file to cluster with command: aws s3 cp s3://'Bucket Name'/HousePrices-0.0.1-SNAPSHOT.jar .
3. Start job with command: spark-submit --class edu.utdallas.Program HousePrices-0.0.1-SNAPSHOT.jar
PS: This method requires uncommenting lines of code pointing input files to S3.
    Also need to repackage project by:
    1. Right-clicking on project nam
    2. Navigating to 'Run As' then 'Maven Build...'
    3. Under Goals enter 'package'.
