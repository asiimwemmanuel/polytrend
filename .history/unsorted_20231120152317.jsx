import React, { useState } from "react";
import "./styles.css"; // Import the external CSS file

const PolyTrendUI = () => {
	const [xValues, setXValues] = useState("");
	const [fXValues, setFXValues] = useState("");
	const [degrees, setDegrees] = useState("");
	const [extrapolates, setExtrapolates] = useState("");
	const [saveToPNG, setSaveToPNG] = useState(false);

	const handleFitPolynomial = () => {
		// Logic for fitting polynomial based on input values
		// (You can implement this based on your specific requirements)
	};

	return (
		<div>
			<h1>Welcome to PolyTrend</h1>
			<p>
				PolyTrend is a regression algorithm that approximates and plots
				a polynomial function onto given data. It provides insights and
				conclusions in the fields of interpolation and polynomial
				regression, specifically in the subfield of approximation
				theory.
			</p>
			<p>
				For csv data, your data MUST be in the form x, f(x). It is
				assumed that values on the same row correspond to the same
				point. Make sure your CSV data starts with the axis labels in
				the first row. Accepted delimiters are whitespace and line
				breaks in the text boxes below.
			</p>
			<div className="input-container">
				{/* Input boxes */}
				<label>X values:</label>
				<input
					type="text"
					value={xValues}
					onChange={(e) => setXValues(e.target.value)}
				/>

				<label>f(X) values:</label>
				<input
					type="text"
					value={fXValues}
					onChange={(e) => setFXValues(e.target.value)}
				/>

				<label>Degrees:</label>
				<input
					type="text"
					value={degrees}
					onChange={(e) => setDegrees(e.target.value)}
				/>

				<label>Extrapolates (optional):</label>
				<input
					type="text"
					value={extrapolates}
					onChange={(e) => setExtrapolates(e.target.value)}
				/>

				{/* Graph area */}
				<div>{/* Graph will change based on input values */}</div>

				{/* Save to PNG checkbox */}
				<label>
					Save to PNG
					<input
						type="checkbox"
						checked={saveToPNG}
						onChange={() => setSaveToPNG(!saveToPNG)}
					/>
				</label>

				{/* Fit Polynomial button */}
				<button onClick={handleFitPolynomial}>Fit Polynomial</button>
			</div>
		</div>
	);
};

export default PolyTrendUI;
