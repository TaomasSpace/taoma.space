import fs from "node:fs/promises";
import path from "node:path";
import { notFound } from "next/navigation";

export const dynamic = "force-dynamic";

type Props = {
  params: { slug?: string[] };
};

const LEGACY_DIR = path.join(process.cwd(), "public", "legacy");

function resolveHtmlPath(slugParts: string[] = []) {
  const candidate = slugParts.join("/") || "index";
  const withExt = candidate.endsWith(".html") ? candidate : `${candidate}.html`;
  const normalized = path.normalize(withExt);
  if (normalized.includes("..")) return null;
  return path.join(LEGACY_DIR, normalized);
}

async function readHtmlFile(filePath: string) {
  try {
    return await fs.readFile(filePath, "utf8");
  } catch {
    return null;
  }
}

export default async function LegacyPage({ params }: Props) {
  const filePath = resolveHtmlPath(params.slug);
  if (!filePath) return notFound();

  const html = await readHtmlFile(filePath);
  if (!html) return notFound();

  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}
