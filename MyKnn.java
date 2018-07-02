package weka.classifiers.lazy;

import java.util.Arrays;

import weka.core.Instance;
import weka.core.pmml.jaxbbindings.NearestNeighborModel;

/**
 * k-NN class to be implemented by you as part of CO3091's coursework2.
 */

public class MyKnn extends KnnParent {

	public MyKnn(int k) {
		super(k);
	}

	public MyKnn() {
		super();
	}

	/* Find K nearest neighbour of a given instance and store them in an array. */
	// Use euclidean distance to find potentially suited neighbours
	@Override
	protected Instance[] findNearestNeighbours(Instance instance) {

		Instance[] nearestNeighbours = new Instance[m_TrainingData.numInstances()]; // tmp ouput array of nearest
																					// instances
		Instance[] tmpNearestNeighbours = new Instance[m_TrainingData.numInstances()];
		Instance[] numberofInstances = new Instance[getK()];

		/* make a for loop for storing instances in training to be modified */
		for (int i = 0; i < m_TrainingData.numInstances(); i++) {
			tmpNearestNeighbours[i] = m_TrainingData.get(i); // This is going to be the instance array that is going to
																// be modified, it is a mirror of m_TrainingData
		}

		/*
		 * This is accounting for the case whereby K is greater than the size of
		 * Training Data
		 */
		if (m_TrainingData.numInstances() < getK()) {
			for (int i = 0; i < nearestNeighbours.length; i++) {
				nearestNeighbours[i] = m_TrainingData.get(i);
			}
			;
			return nearestNeighbours;
		}
		/* This is the alternative where the training data is larger than K */
		if (m_TrainingData.numInstances() > getK()) {

			// Bubble sort for sorting instances based on Euclidian distance
			for (int i = 0; i < tmpNearestNeighbours.length; i++) { // Want to compare instance parameter with each
																	// instance in m_TrainingData

				for (int j = 1; j < tmpNearestNeighbours.length - i; j++) {
					if (euclideanDistance(tmpNearestNeighbours[j - 1],
							instance) > euclideanDistance(tmpNearestNeighbours[j], instance)) { // Calculating and
																								// comparing the
																								// Euclidian distance of
																								// each Instance in the
																								// array

						Instance temp = tmpNearestNeighbours[j - 1];
						tmpNearestNeighbours[j - 1] = tmpNearestNeighbours[j];
						tmpNearestNeighbours[j] = temp;
					}
				}
			}
			/* Return number of instances equal to k */
			for (int a = 0; a < numberofInstances.length; a++) {
				numberofInstances[a] = tmpNearestNeighbours[a];
			}
		}

		return numberofInstances;

	}

	/*
	 * This method normalises all input attributes of m_TrainingData. This is done
	 * by using a for loop and traversing m_TrainingData.
	 */
	@Override
	protected void normaliseNumericInputAttributesTrainingData() {
		int numberOfInstancesInTrainingData = m_TrainingData.numInstances();

		/* Traverse the individual instances in m_TrainingData */
		for (int i = 0; i < numberOfInstancesInTrainingData; i++) {
			
			/*
			 * Recursively run normaliseNumericInputAttributes on by passing each instance
			 * available in the training Data in order to normalise the data.
			 */
			normaliseNumericInputAttributes(m_TrainingData.instance(i));

		}

	}

	/*
	 * Normalise input attributes of a single instance using normalisation formula.
	 * This insures attributes of all data fall between 0 and 1.
	 */
	@Override
	protected void normaliseNumericInputAttributes(Instance instance) {

		/* Traversing the input attributes in a given instance. */
		for (int i = 0; i < m_TrainingData.numAttributes(); i++) {
			/*
			 * Here we check if the attribute at index I is not an output attribute since we
			 * are only normalising NUMERICAL INPUT attributes
			 */
			if (i != instance.classIndex() && instance.attribute(i).isNumeric()) {

				/* Formula for normalising the attribute at index i */
				double normaliseI = (instance.value(i) - min[i]) / (max[i] - min[i]);

				/* Set the new normalised input attributes */
				instance.setValue(i, normaliseI);

			}

		}
	}

	/*
	 * This method sets the min and max arrays and fills them with each respective
	 * possible value for each numerical input attribute of m_TrainingData
	 */
	@Override
	protected void determineMinMaxAttributeValues() {

		min = new double[m_TrainingData.numAttributes()]; // Array storing min values for m_TrainingData
		max = new double[m_TrainingData.numAttributes()]; // Array storing max values for m_TrainingData

		for (int i = 0; i < m_TrainingData.numInstances(); i++) {
			for (int j = 0; j < m_TrainingData.instance(i).numAttributes(); j++) {

				/*
				 * First we check if the value at index position i in the iteration is equal to
				 * 0 (array was instantiated to be filled with 0's
				 */ if (i == 0) {
					min[j] = m_TrainingData.instance(i).value(j);
					max[j] = m_TrainingData.instance(i).value(j);
				} else {
					/*
					 * Else we are comparing the values at a given instance with its value at index
					 * j and assigning it to the index in the min or max array based on whether it
					 * is greater than or smaller than the value at the index in the min or max
					 * array.
					 */
					if (m_TrainingData.instance(i).value(j) < min[j]) {
						min[j] = m_TrainingData.instance(i).value(j);
					}
					if (m_TrainingData.instance(i).value(j) > max[j]) {
						max[j] = m_TrainingData.instance(i).value(j);
					}

				}
			}
		}
	}

	/*
	 * This method determines the distance between the input attributes of two
	 * instances
	 */
	@Override
	protected double euclideanDistance(Instance instance1, Instance instance2) {

		double result = 0;
		for (int index = 0; index < instance1.numValues(); index++) {
			if (instance1.attribute(index).isNumeric() && index != m_TrainingData.classIndex()) { // Here we are
																									// checking that the
																									// attribute is
																									// numerical and
																									// that it is not
																									// equal to the
																									// output attribute.
				result += (Math.pow(instance1.value(index) - instance2.value(index), 2));

				// This else if case evaluates for the instance where the attribute is not
				// Numerical (i.e. categorical)
			} else if (instance1.attribute(index).isNominal() && index != m_TrainingData.classIndex()) {
				if (instance1.value(index) == instance2.value(index)) { // Here we are checking if both categorical
																		// attributes are the same, if they are then we
																		// add 0 to the result, if not we add 1
					result += 0;
				} else {
					result += 1;
				}
			}

		}

		return Math.sqrt(result);
	}

	/*
	 * Need to determine the average of outputs gathered from nearestNeighbours,
	 * where average = mean. Formula = sum of output from each neighbour / total
	 * number of neighbours classValue is used to extract the numerical value of the
	 * output attribute for any neighbour
	 */
	@Override
	public double determinePredictedOutput(Instance[] nearestNeighbours) {

		double totalOutput = 0;
		for (int i = 0; i < nearestNeighbours.length; i++) {
			totalOutput += nearestNeighbours[i].classValue();
		}
		return totalOutput / nearestNeighbours.length;
	}

}
