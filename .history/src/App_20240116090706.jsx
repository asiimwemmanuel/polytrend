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
		<div
			className="App"
			style={{
				display: "flex",
				flexDirection: "column",
				alignItems: "center",
				justifyContent: "center",
				padding: "10px",
			}}
		>
			<h1>Welcome to PolyTrend 📈</h1>
			<p>
				PolyTrend - a simple clean way to create polynomial fits in data
			</p>
			<div
				style={{
					flexDirection: "column",
					display: "flex",
					justifyContent: "flex-start",
					alignItems: "flex-start",
				}}
			>
				<p style={{ textAlign: "left" }}>
					For csv data, your data MUST be in the form x, f(x). <br />{" "}
					It is assumed that values on the same row correspond to the
					same point.
					<br /> Make sure your CSV data starts with the axis lables
					in the first row
					<br /> Accepted delimiters are whitespace and linebreaks in
					the text boxes below
					<br /> Saved PNGs will be in the 'images' folder of the
					application
					<br /> For the algebraic expressions of the models, visit
					the 'log' folder
				</p>
			</div>
			<div style={{ display: "flex", flexDirection: "row" }}>
				<div
					className="input-label-pair"
					style={{
						display: "flex",
						flexDirection: "row",
						alignItems: "center",
						justifyContent: "center",
						marginRight: "20px",
					}}
				>
					<label style={{ marginRight: "10px" }}>x values</label>
					{/* <textarea name="ggg" id="" cols="30" rows="10" about="g" placeholder="Input your x values..."></textarea> */}
					<textarea
						placeholder="Input your x values..."
						style={{
							height: "350px",
							width: "100px",
							backgroundColor: "white",
							fontSize: "15px",
							// textAlign: 'start',
							color: "black",
							resize: "none",
						}}
					></textarea>
				</div>
				<div
					className="input-label-pair"
					style={{
						display: "flex",
						flexDirection: "row",
						alignItems: "center",
						justifyContent: "center",
						marginRight: "20px",
					}}
				>
					<label style={{ marginRight: "10px" }}>f(x) values</label>
					<textarea
						placeholder="Input your y values..."
						style={{
							height: "350px",
							backgroundColor: "white",
							fontSize: "15px",
							width: "100px",
							color: "black",
							resize: "none",
						}}
					></textarea>
				</div>
				<div
					className="input-label-pair"
					style={{
						display: "flex",
						flexDirection: "row",
						alignItems: "center",
						justifyContent: "center",
						marginRight: "20px",
					}}
				>
					<label style={{ marginRight: "10px" }}>Degrees</label>
					<textarea // textAlign: 'start',
						placeholder="Specify input degrees..."
						style={{
							height: "350px",
							width: "100px",
							backgroundColor: "white",
							resize: "none",
							fontSize: "15px",
							// textAlign: 'start',
							color: "black",
							wordWrap: "break-word",
						}}
					></textarea>
				</div>
				<div
					className="input-label-pair"
					style={{
						display: "flex",
						flexDirection: "row",
						alignItems: "center",
						justifyContent: "center",
						marginRight: "20px",
					}}
				>
					<label style={{ marginRight: "10px" }}>
						forecasts <br /> (optional)
					</label>
					<textarea
						placeholder="Input data for model prediction..."
						style={{
							height: "350px",
							backgroundColor: "white",
							resize: "none",
							fontSize: "15px",
							width: "100px",
							color: "black",
						}}
					></textarea>
				</div>
			</div>
			<div
				style={{
					display: "flex",
					flexDirection: "column",
					justifyContent: "center",
					alignItems: "center",
				}}
			>
				<button
					style={{
						marginTop: "20px",
						backgroundColor: "#ff8800",
						width: "750px",
					}}
				>
					<b>Import csv file</b>
				</button>
				<button
					style={{
						marginTop: "20px",
						backgroundColor: "#ff8800",
						width: "750px",
					}}
				>
					Fit polynomial
				</button>
			</div>
			<div
				style={{
					display: "flex",
					flexDirection: "row",
					alignItems: "center",
					justifyContent: "center",
					padding: "10px",
				}}
			>
				<input type="checkbox" name="STPNG" id="checkbox" />
				<p> Save to PNG</p>
			</div>
		</div>
	);
}

export default App;
