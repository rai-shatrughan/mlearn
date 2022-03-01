package dl.preprocess;

import java.io.File;
import java.io.FileWriter;

import java.util.List;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

public class Imputation {

    static String filepath = "src/test/resources/data";
    static String filename = "house_tiny.csv";
    static Table inputs, outputs;

    public static void main(String[] args) {
        createData();
        Impute();
        createNDArray();
    }

    private static void createNDArray() {
        NDManager nd = NDManager.newBaseManager();
        NDArray x = nd.create(inputs.as().doubleMatrix());
        NDArray y = nd.create(outputs.as().intMatrix());
        System.out.println(x);
        System.out.println(y);
    }

    private static void Impute() {
        try {
            Table data = Table.read().file(filepath + filename);
            // System.out.println(data);
            inputs = Table.create(data.columns());
            inputs.removeColumns("Price");
            outputs = data.select("Price");

            Column colNumRoom = inputs.column("NumRooms");
            colNumRoom.set(colNumRoom.isMissing(), (int) inputs.nCol("NumRooms").mean());

            StringColumn colAlley = (StringColumn) inputs.column("Alley");
            List<BooleanColumn> dummies = colAlley.getDummies();
            inputs.removeColumns(colAlley);
            inputs.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                    DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray()));

            System.out.println(inputs);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void createData() {
        try {

            File file = new File(filepath);
            file.mkdir();

            String dataFile = filepath + filename;

            // Create file
            File f = new File(dataFile);
            f.createNewFile();

            // Write to file
            try (FileWriter writer = new FileWriter(dataFile)) {
                writer.write("NumRooms,Alley,Price\n"); // Column names
                writer.write("NA,Pave,127500\n"); // Each row represents a data example
                writer.write("2,NA,106000\n");
                writer.write("4,NA,178100\n");
                writer.write("NA,NA,140000\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
