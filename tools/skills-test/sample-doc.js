const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign, LevelFormat, PageNumber, Footer, Header, ExternalHyperlink } = require('docx');
const fs = require('fs');

// Create the document
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } }, // 12pt default
    paragraphStyles: [
      // Override built-in Title style
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 56, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER } },
      // Override built-in heading styles
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 180, after: 180 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-list",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }, // 1 inch margins
        size: { orientation: 'portrait' },
        pageNumbers: { start: 1, formatType: "decimal" }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ 
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "Sample Document", size: 20 })]
      })] }) 
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ 
        alignment: AlignmentType.CENTER,
        children: [new TextRun("Page "), new TextRun({ children: [PageNumber.CURRENT] }), new TextRun(" of "), new TextRun({ children: [PageNumber.TOTAL_PAGES] })]
      })] }) 
    },
    children: [
      // Title
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Sample Word Document")] }),
      
      // Author information
      new Paragraph({ 
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Created with docx.js", size: 20, italics: true })]
      }),
      
      // Introduction
      new Paragraph({ 
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("1. Introduction")]
      }),
      
      new Paragraph({ 
        children: [new TextRun("This is a sample Word document created using the docx.js library. It demonstrates various document elements including headings, paragraphs, lists, tables, and links.")]
      }),
      
      new Paragraph({ 
        children: [new TextRun("The docx.js library allows for creating professional-looking Word documents programmatically using JavaScript or TypeScript. This document showcases some of the key features available.")]
      }),
      
      // Features section
      new Paragraph({ 
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("2. Key Features")]
      }),
      
      // Bullet list
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Text formatting (bold, italic, underline)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Heading styles and document structure")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Bullet and numbered lists")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Tables with headers and borders")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("External links")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Headers, footers, and page numbers")] }),
      
      // Example table
      new Paragraph({ 
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("3. Example Table")]
      }),
      
      new Paragraph({ 
        children: [new TextRun("Below is an example table demonstrating table formatting in docx.js:")]
      }),
      
      // Table creation
      new Table({
        columnWidths: [4680, 4680], // Two equal columns
        margins: { top: 100, bottom: 100, left: 180, right: 180 },
        rows: [
          new TableRow({
            tableHeader: true,
            children: [
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ 
                  alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "Product Name", bold: true, size: 22 })]
                })]
              }),
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ 
                  alignment: AlignmentType.CENTER,
                  children: [new TextRun({ text: "Price", bold: true, size: 22 })]
                })]
              })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun("Laptop")] })]
              }),
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun("$1,299")] })]
              })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun("Smartphone")] })]
              }),
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun("$799")] })]
              })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun("Tablet")] })]
              }),
              new TableCell({
                borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, 
                           bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                           right: { style: BorderStyle.SINGLE, size: 1, color: "000000" } },
                width: { size: 4680, type: WidthType.DXA },
                children: [new Paragraph({ children: [new TextRun("$499")] })]
              })
            ]
          })
        ]
      }),
      
      // Numbered list
      new Paragraph({ 
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("4. Numbered List Example")]
      }),
      
      new Paragraph({ numbering: { reference: "numbered-list", level: 0 },
        children: [new TextRun("First item in a numbered list")] }),
      
      new Paragraph({ numbering: { reference: "numbered-list", level: 0 },
        children: [new TextRun("Second item with more detailed information")] }),
      
      new Paragraph({ numbering: { reference: "numbered-list", level: 0 },
        children: [new TextRun("Third item demonstrating list continuation")] }),
      
      // Links section
      new Paragraph({ 
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("5. External Links")]
      }),
      
      new Paragraph({ 
        children: [new TextRun("You can include external hyperlinks in your document:")]
      }),
      
      new Paragraph({ 
        children: [
          new TextRun("Visit the docx.js GitHub repository: "),
          new ExternalHyperlink({
            children: [new TextRun({ text: "docx.js GitHub", style: "Hyperlink" })],
            link: "https://github.com/dolanmiu/docx"
          })
        ]
      }),
      
      // Conclusion
      new Paragraph({ 
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("6. Conclusion")]
      }),
      
      new Paragraph({ 
        children: [new TextRun("This sample document demonstrates the capabilities of the docx.js library for creating professional Word documents programmatically. The library provides extensive support for various document elements and formatting options.")]
      }),
      
      new Paragraph({ 
        children: [new TextRun("For more information and documentation, please visit the official docx.js website or GitHub repository.")]
      })
    ]
  }]
});

// Generate and save the document
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("sample-document.docx", buffer);
  console.log("Document generated successfully: sample-document.docx");
}).catch(err => {
  console.error("Error generating document:", err);
});
