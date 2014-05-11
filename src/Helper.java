import java.util.Vector;

import org.apache.hadoop.io.Text;

public class Helper {

	public static String getPoint(Vector<Double> pt, int dim) {
		if (pt == null)
			return "";

		String ans = "";
		for (int i = 0; i < dim; i++)
			ans += pt.get(i) + ",";

		return ans.substring(0, ans.length() - 1);
	}

	public static double calculateDistance(Vector<Double> p1,
			Vector<Double> p2, int dim) {
		double sum = 0.0;
		for (int i = 0; i < dim; i++)
			sum += (p1.get(i) - p2.get(i)) * (p1.get(i) - p2.get(i));
		return Math.sqrt(sum);
	}

	public static Vector<Double> parsePoint(Text line, int dim) {
		Vector<Double> pt = new Vector<Double>(dim);
		String[] parts = line.toString().split(",");
		for (int i = 0; i < dim; i++)
			pt.add(Double.parseDouble(parts[i]));
		return pt;
	}
}
