// import path = require("path");
// import fs = require("fs-extra");
// import jsonfile = require("jsonfile");
// import axios from "axios";

// const callGrobid = async () => {
//   //   const pdfData = await fs.readFile(
//   //     "F:\\GoogleSync\\pdfs\\The Economies and Dimensionality of Design Prototyping Value, Time, Cost, and Fidelit.pdf"
//   //   );

//   //   const data = await axios.post(
//   //     "http://52.10.103.106/autograb/grobidmetadata",
//   //     pdfData,
//   //     {
//   //       headers: { "Content-Type": "application/pdf" }
//   //     }
//   //   );

//   //   console.log(data);
//   var params = {
//     // Request parameters
//     expr: "W='testing'",
//     model: "latest",
//     attributes: "Ti",
//     count: "10",
//     offset: "0"
//   };
//   //      "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?expr=And(W='poverty',W='impedes')&model=latest&count=10&offset=0&attributes=Ti",
//   // "https://api.labs.cognitive.microsoft.com/academic/v1.0/interpret?query=poverty impedes&complete=0&count=10&model=latest",
// const url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?"

//   try {
//     const ms = await axios.get(
//       url + "expr=Ti='active learning increases student performance in science engineering and mathematics'&model=latest&count=10&offset=0&attributes=E.DN,E.VFN,Ti,Y,D,ECC,AA.AuN,W,VSN,IA,",
//       {
//         // params,
//         headers: {
//           Host: "api.labs.cognitive.microsoft.com",
//           "Ocp-Apim-Subscription-Key": "2681b4126f064fa2a7bbf223a6b10734"
//         }
//       }
//     );
//     console.log(ms.data.entities[0]);
//   } catch (err) {
//     console.log(err);
//   }
// };
// callGrobid();
// //@ts-nocheck
// import path = require("path");
// import jsonfile = require("jsonfile");
// import Plain from "slate-plain-serializer";
// import convertBase64 from "slate-base64-serializer";
// import { PdfSegmentViewbox, UserDoc } from "../creators";

// const convertStateJson = async () => {
//   let state = await jsonfile.readFile(
//     "C:\\Users\\merha\\Desktop\\joel\\state1.json"
//   );
//   Object.values(state.graph.nodes).forEach(
//     (node: PdfSegmentViewbox | UserDoc) => {
//       /// CONVERT STLYE FOR VIEWBOX

//       const { id } = node as PdfSegmentViewbox | UserDoc;
//       const { left, top, width, height } = node.style;
//       if (node.data.type === "pdf.segment.viewbox") {
//         const newStyle = {
//           min: {
//             id,
//             left,
//             top,
//             width: 220,
//             height: 60
//           },
//           max: {
//             id,
//             left,
//             top,
//             width,
//             height
//           },
//           modes: ["min", "max"],
//           modeIx: 0,
//           lockedCorner: "nw"
//         };
//         state.graph.nodes[id].style = newStyle;
//       }

//       if (node.data.type === "userHtml") {
//         const { text } = node.data;
//         const newData = {
//           type: "userDoc",
//           base64: convertBase64.serialize(Plain.deserialize(text)),
//           text,
//           useTextForAutocomplete: true
//         };

//         const newStyle = {
//           min: {
//             id,
//             left,
//             top,
//             width,
//             height
//           },
//           max: {
//             id,
//             left,
//             top,
//             width,
//             height
//           },
//           modes: ["min", "max"],
//           modeIx: 0,
//           lockedCorner: "nw",
//           fontSize: 26
//         };
//         state.graph.nodes[id].style = newStyle;
//         state.graph.nodes[id].data = newData;
//       }
//     }
//   );

//   await jsonfile.writeFile(
//     "C:\\Users\\merha\\Desktop\\joel\\state.json",
//     state
//   );
// };
// convertStateJson();
// // "id": "334059b0-66e5-11e9-b020-43702fcef17c",
// // "data": {
// //   "type": "userDoc",
// //   "base64": "JTdCJTIyb2JqZWN0JTIyJTNBJTIydmFsdWUlMjIlMkMlMjJkb2N1bWVudCUyMiUzQSU3QiUyMm9iamVjdCUyMiUzQSUyMmRvY3VtZW50JTIyJTJDJTIyZGF0YSUyMiUzQSU3QiU3RCUyQyUyMm5vZGVzJTIyJTNBJTVCJTdCJTIyb2JqZWN0JTIyJTNBJTIyYmxvY2slMjIlMkMlMjJ0eXBlJTIyJTNBJTIycGFyYWdyYXBoJTIyJTJDJTIyZGF0YSUyMiUzQSU3QiU3RCUyQyUyMm5vZGVzJTIyJTNBJTVCJTdCJTIyb2JqZWN0JTIyJTNBJTIydGV4dCUyMiUyQyUyMmxlYXZlcyUyMiUzQSU1QiU3QiUyMm9iamVjdCUyMiUzQSUyMmxlYWYlMjIlMkMlMjJ0ZXh0JTIyJTNBJTIyYXNkZiUyMiUyQyUyMm1hcmtzJTIyJTNBJTVCJTVEJTdEJTVEJTdEJTVEJTdEJTVEJTdEJTdE",
// //   "text": "asdf",
// //   "useTextForAutocomplete": true
// // },
// // "meta": {
// //   "createdBy": "defaultUser",
// //   "timeCreated": 1556147012326,
// //   "timeUpdated": 1556147026096
// // },
// // "style": {
// //   "min": {
// //     "left": 30,
// //     "top": 50,
// //     "width": 220,
// //     "height": 120
// //   },
// //   "max": {
// //     "left": 121,
// //     "top": 26,
// //     "width": 220,
// //     "height": 120
// //   },
// //   "modes": [
// //     "max",
// //     "min"
// //   ],
// //   "modeIx": 0,
// //   "lockedCorner": "nw",
// //   "fontSize": 26
// // }
// // "c76fd820-6129-11e9-a71a-0d16ecaabc88": {
// //     "id": "c76fd820-6129-11e9-a71a-0d16ecaabc88",
// //     "data": {
// //       "type": "userHtml",
// //       "html": "<p>lots of problems with biomed scholarly communication and scientific work</p>",
// //       "text": "lots of problems with biomed scholarly communication and scientific work"
// //     },
// //     "meta": {
// //       "createdBy": "defaultUser",
// //       "timeCreated": 1555277191234,
// //       "timeUpdated": 1555516786517
// //     },
// //     "style": {
// //       "left": 2615.0380935149865,
// //       "top": 2538.2004383356802,
// //       "width": 300,
// //       "height": 110
// //     }
// //   }

// //     "style": {
// //       "id": "00af34e0-6149-11e9-a71a-0d16ecaabc88",
// //       "type": "circle",
// //       "left": 149.70194919935105,
// //       "top": 1883.0035507527982,
// //       "width": 605.8286,
// //       "height": 180.22800000000007,
// //       "fill": "blue",
// //       "draggabled": true,
// //       "radius": 5,
// //       "stroke": "blue",
// //       "strokeWidth": 4
// //     }
// //   }

// //     "style": {
// //       "min": {
// //         "id": "32adb6a0-66e5-11e9-b020-43702fcef17c",
// //         "left": 30,
// //         "top": 170,
// //         "width": 220,
// //         "height": 60
// //       },
// //       "max": {
// //         "id": "32adb6a0-66e5-11e9-b020-43702fcef17c",
// //         "left": 30,
// //         "top": 170,
// //         "width": 583.866744832,
// //         "height": 191.65519999999992
// //       },
// //       "modes": [
// //         "min",
// //         "max"
// //       ],
// //       "modeIx": 0,
// //       "lockedCorner": "nw"
// //     }

// //   await jsonfile.writeFile(textToDisplayFile, {
// //     pageNumber,
// //     text: textToDisplay,
// //     // maybe use this to detect rotation?
// //     viewportFlat: { width, height, xMin, yMin, xMax, yMax }
// //   });
