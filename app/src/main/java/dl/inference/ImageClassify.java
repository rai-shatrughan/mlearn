package dl.inference;

import java.awt.image.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import ai.djl.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.translate.*;

public final class ImageClassify {

    public static void main(String[] args) throws Exception{
        Image img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();

        Path modelDir = Paths.get("build/mlp");
        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
        model.load(modelDir);

        Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Convert Image to NDArray
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }
        
            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // Create a Classifications with the output probabilities
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }
        
            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };

        Predictor predictor = model.newPredictor(translator);

        Object classifications = predictor.predict(img);

        System.out.println(classifications.toString());

    }
    
}