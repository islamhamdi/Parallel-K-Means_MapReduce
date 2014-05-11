import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.Collections;
import java.util.Vector;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/*This class is responsible for running map reduce job*/
public class MapReduceDriver extends Configured implements Tool {
	public static final int DIMENSTION = 4;
	public static final int K_CENTROIDS = 3;

	public static String inputDataPath = "bezdekIris.data.txt";
	public static String centroidsPath = "centroids.txt";
	public static String outputPath = "output/";
	public static String outputfile = "/part-r-00000";
	public static final double ALPHA = 1e-3;
	public static boolean done = false, flag = true;

	public static Vector<Vector<Double>> centers = new Vector<Vector<Double>>(
			K_CENTROIDS);

	public void configure(Configuration conf, int itr) {
		centers.clear();
		try {
			URI[] cacheFiles = DistributedCache.getCacheFiles(conf);
			if (cacheFiles != null && cacheFiles.length > 0) {
				BufferedReader br = new BufferedReader(new FileReader(
						cacheFiles[itr].toString()));
				Vector<Vector<Double>> list = new Vector<Vector<Double>>();
				try {
					String line = br.readLine();
					while (line != null) {
						String[] parts = line.split(",");
						Vector<Double> value = new Vector<Double>(DIMENSTION);
						for (int i = 0; i < DIMENSTION; i++)
							value.add(Double.parseDouble(parts[i]));
						list.add(value);
						line = br.readLine();
					}
				} finally {
					br.close();
				}
				Collections.shuffle(list);
				for (int i = 0; i < K_CENTROIDS; i++)
					centers.add(list.get(i));
			}
		} catch (IOException e) {
			System.err.println("Exception reading DistribtuedCache: " + e);
		}
	}

	public static class KMeansMapper extends
			Mapper<LongWritable, Text, Text, Text> {

		// Maps (key, point) --> (center, point)
		@Override
		public void map(LongWritable center_key, Text point_value,
				Context context) throws IOException, InterruptedException {

			Vector<Double> point_val = Helper.parsePoint(point_value,
					DIMENSTION);

			Vector<Double> nearestCenter = null;
			double nearestDistance = Double.MAX_VALUE;
			for (int i = 0; i < centers.size(); i++) {
				double dist = Helper.calculateDistance(point_val,
						centers.get(i), DIMENSTION);
				if (dist < nearestDistance) {
					nearestDistance = dist;
					nearestCenter = centers.get(i);
				}
			}
			context.write(new Text(Helper.getPoint(nearestCenter, DIMENSTION)),
					new Text(Helper.getPoint(point_val, DIMENSTION)));
		}
	}

	public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {

		// (oldCenter, cluster pts) --> (newCenter, new cluster pts)
		@Override
		public void reduce(Text center, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {

			Vector<Double> newCenter = new Vector<Double>(DIMENSTION); // 0's

			for (int i = 0; i < DIMENSTION; i++)
				newCenter.add(0.0);

			int clusterSize = 0;
			String candidatePoints = "";
			for (Text curPoint : values) {
				clusterSize++;
				candidatePoints += ",\t\t" + curPoint.toString();

				Vector<Double> pts = Helper.parsePoint(curPoint, DIMENSTION);
				for (int i = 0; i < DIMENSTION; i++)
					newCenter.set(i, newCenter.get(i) + pts.get(i));
			}
			for (int i = 0; i < DIMENSTION; i++)
				newCenter.set(i, newCenter.get(i) / clusterSize);

			flag &= Helper.calculateDistance(newCenter,
					Helper.parsePoint(center, 4), 4) < ALPHA;

			context.write(new Text(Helper.getPoint(newCenter, DIMENSTION)),
					new Text(candidatePoints));
			System.out.println("Cluster Size = " + clusterSize);
		}
	}

	public int run(String[] args) throws Exception {
		int itr = 0;
		String outputPa = outputPath + itr;
		String prevOutputPa = "";
		long start = System.nanoTime();
		while (!done) {
			flag = true;

			System.out.println("Start");
			Configuration conf = getConf();
			Job job = new Job(conf, "ParallelKMeans");

			if (itr == 0) {
				Path hdfsPath = new Path(centroidsPath);
				// upload the file to hdfs. Overwrite any existing copy.
				DistributedCache.addCacheFile(hdfsPath.toUri(), conf);
			} else {
				Path hdfsPath = new Path(prevOutputPa + outputfile);
				// upload the file to hdfs. Overwrite any existing copy.
				DistributedCache.addCacheFile(hdfsPath.toUri(), conf);
			}

			job.setMapperClass(KMeansMapper.class);
			job.setReducerClass(KMeansReducer.class);

			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Text.class);

			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);

			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);

			FileInputFormat.addInputPath(job, new Path(inputDataPath));
			FileOutputFormat.setOutputPath(job, new Path(outputPa));

			configure(conf, itr);
			// clearOutputs(outputPath);
			job.waitForCompletion(true);
			itr++;
			outputPa = outputPath + itr;
			prevOutputPa = outputPath + (itr - 1);

			done |= flag;
		}
		long end = System.nanoTime();
		System.out.println((end - start) * 1e-9);
		return 0;
	}

	public static void clearOutputs(String string) throws IOException {
		FileUtils.deleteDirectory(new File(string));
	}

	public static void main(String[] args) throws Exception {
		MapReduceDriver driver = new MapReduceDriver();
		ToolRunner.run(driver, args);
	}

}