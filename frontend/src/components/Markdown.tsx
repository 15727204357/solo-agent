import type { ReactNode } from "react";

type MarkdownProps = {
  content: string;
};

export function Markdown({ content }: MarkdownProps) {
  if (!content.trim()) {
    return <p className="text-sm text-slate-400">等待输出...</p>;
  }

  const nodes: ReactNode[] = [];
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  let paragraph: string[] = [];
  let code: string[] = [];
  let inCodeBlock = false;

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }
    nodes.push(
      <p key={`p-${nodes.length}`} className="whitespace-pre-wrap leading-7">
        {paragraph.join("\n")}
      </p>,
    );
    paragraph = [];
  };

  lines.forEach((line) => {
    if (line.startsWith("```")) {
      if (inCodeBlock) {
        nodes.push(
          <pre key={`code-${nodes.length}`} className="overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-100">
            <code>{code.join("\n")}</code>
          </pre>,
        );
        code = [];
        inCodeBlock = false;
      } else {
        flushParagraph();
        code = [];
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      code.push(line);
      return;
    }

    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      const className = level === 1 ? "text-xl font-semibold" : level === 2 ? "text-lg font-semibold" : "font-semibold";
      nodes.push(
        <h3 key={`h-${nodes.length}`} className={className}>
          {heading[2]}
        </h3>,
      );
      return;
    }

    const listItem = /^[-*]\s+(.*)$/.exec(line);
    if (listItem) {
      flushParagraph();
      nodes.push(
        <div key={`li-${nodes.length}`} className="flex gap-2 leading-7">
          <span className="mt-3 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400" />
          <span>{listItem[1]}</span>
        </div>,
      );
      return;
    }

    if (!line.trim()) {
      flushParagraph();
      return;
    }

    paragraph.push(line);
  });

  flushParagraph();

  if (inCodeBlock) {
    nodes.push(
      <pre key={`code-${nodes.length}`} className="overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-100">
        <code>{code.join("\n")}</code>
      </pre>,
    );
  }

  return <div className="space-y-3 text-sm text-slate-700 dark:text-slate-200">{nodes}</div>;
}
