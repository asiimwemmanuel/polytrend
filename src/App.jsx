import { save } from "@tauri-apps/api/dialog";
import { useState } from "react";
import "./App.css";

function App() {
	const [name, setName] = useState("");

	// define object for file stuff
	const saveImage = async (dataUrl) => {
		const suggestedFilename = "image.png";

		// Save into the default downloads directory, like in the browser
		const filePath = await save({
			defaultPath: (await downloadsDir()) + "/" + suggestedFilename,
		});

		// Now we can write the file to the disk
		await writeBinaryFile(
			file,
			await fetch(dataUrl).then((res) => res.blob())
		);
	};

	return (
		<div className="container">
			<h1>Welcome to PolyTrend! 📈</h1>
			{/* make this a subheading */}
			<p>A simple clean way to create lines of best fit on data</p>
			{/* guidlelines */}
			<ul>
				<li>
					CSV data must be in the form x,f(x) and starting with axis
					labels
				</li>
				<li>Values on the same row correspond to the same point</li>
				<li>
					Accepted delimiters are whitespace(?) and linebreaks(?) in
					the text boxes below
				</li>
			</ul>
			{/* make sure to add a section on the view for the algebraic expressions */}
			{/* add an empty graph that updates when the fit polynomial button is pressed */}
			{/* TODO: format for the boxes */}
			<form
				className="row"
				onSubmit={(e) => {
					e.preventDefault();
					greet();
				}}
			>
				<label htmlFor="x-values">x values</label>
				<textarea
					id="greet-input"
					rows={7}
					onChange={(e) => setName(e.currentTarget.value)}
					placeholder="Input your x values..."
				/>
				<label htmlFor="x-values">f(x) values</label>
				<textarea
					id="greet-input"
					rows={7}
					onChange={(e) => setName(e.currentTarget.value)}
					placeholder="Input your y values..."
				/>
				<label htmlFor="degrees">degrees</label>
				<textarea
					id="greet-input"
					rows={7}
					onChange={(e) => setName(e.currentTarget.value)}
					placeholder="Specify polynomial degrees..."
				/>
				<textarea
					id="greet-input"
					rows={7}
					onChange={(e) => setName(e.currentTarget.value)}
					placeholder="Input data for model prediction"
				/>
				<button>Import CSV file</button>
				<button type="submit">Fit polynomial</button>
			</form>
		</div>
	);
}

export default App;
